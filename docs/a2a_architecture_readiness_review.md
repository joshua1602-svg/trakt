# Trakt A2A Architecture Review and Gap Analysis

**Status:** Read-only architecture review. No production code was modified.
**Date:** 2026-08-11
**Branch:** `claude/trakt-a2a-architecture-review-ogm81v`
**Question asked:** how close is the existing Trakt architecture to becoming
A2A-native — usable by humans through the current UI *and* by external AI agents
(Claude / OpenAI / Copilot / a client's own agent) against governed, deterministic
credit functions, with permissions, provenance, workflow state and audit as
first-class facts?

**Method:** repository inspection and end-to-end flow tracing against the actual
implementations, not filenames. Approximately 250k lines of Python across
`engine/`, `mi_agent/`, `mi_agent_api/`, `operations_control/`, `trakt_core/`,
`analytics/`, `simulation/`, plus two React frontends, 65 YAML configuration
files and 225 test modules.

---

## 1. Executive assessment

### How A2A-ready is Trakt today?

**Indicative readiness: 45%.**

That number is deliberately uncomfortable, and it hides an unusually favourable
shape. Trakt has already built the parts that are normally the *hardest* and
slowest to retrofit — a canonical credit representation, a deterministic
calculation layer that an LLM cannot bypass, an interface-neutral governance
core with an explicit organisation × resource × capability entitlement model, a
result envelope that already carries policy and provenance, and hash-chained
audit. What it has *not* built is the part that is normally cheap: a typed,
agent-callable tool surface and a machine identity to call it with.

That is the right way round. Most credit platforms attempting this have the
opposite problem — a REST API an agent can call, wrapping business logic that
lives in reports, with no provenance and no permission model. Trakt's remaining
work is largely additive exposure of things that already exist.

Broken down:

| Dimension | Readiness | Why |
|---|---:|---|
| Canonical credit representation | 85% | 499-field registry + 242-field business-semantics layer + mapping/alias tiers. Entities/relationships implicit, not declared. |
| Deterministic credit primitives (implementation) | 80% | Concentration/covenant engine with 24 registered evaluators, period-change engine, bridge, cohorts, stratification, validation rules, forecast. |
| Deterministic primitives *exposed as tools* | 15% | Exactly one governed capability an agent can call meaningfully (`mi.question.answer`) and it takes prose, not typed arguments. |
| Identity / permissions (model) | 75% | `trakt_core` implements organisations, principals, resources, grants, scopes, and it is tested. |
| Identity / permissions (in service) | 20% | Every config file is `*.example.yaml`. Entitlements are dormant. No machine identity except the Copilot Entra path. |
| Provenance available to an agent | 40% | Dataset/snapshot-level provenance is in the envelope today. Field- and value-level lineage exists as pipeline artefacts but is not served. |
| Workflow state for A2A | 30% | Two strong state machines exist (OCC delivery, onboarding case) and an `InformationRequest` object that is 70% of a DD request — but scoped to onboarding, not to a portfolio. |
| Audit of agent actions | 50% | One structured audit line per governed execution; hash-chained audit exists but only in the OCC store. Not queryable per agent. |
| Model-agnosticism | 70% | The LLM sits behind a `Protocol`, only proposes a validated spec, and the deterministic path needs no LLM at all. Concrete adapters are Anthropic-only today. |
| External interface | 25% | One hand-authored OpenAPI 3.0.3 with three operations, built for M365 Copilot. No MCP. No general agent interface. |

### What existing architecture is particularly reusable?

Five things, and they carry most of the weight:

1. **`trakt_core/` (5,415 lines).** An interface-neutral governance core with a
   hard rule — no FastAPI, no Azure, no pandas — enforced by
   `tests/test_governance_dependency_direction.py`, including a subprocess check
   that *calling* a capability never loads a web framework. It already defines
   `CHANNEL_ENTERPRISE_AGENT` and `CHANNEL_AGENT_TO_AGENT`
   (`trakt_core/context.py:39-40`). An MCP server or an agent REST surface is a
   new adapter over this, not a new architecture.

2. **`GovernedResult[T]` (`trakt_core/envelope.py`).** Already carries
   `capability`, `schema_version`, `status`, `request_id`, `correlation_id`,
   `tenant_id`, `portfolio_id`, `SnapshotRef`, `ScopeRef`, `PolicyState`,
   `ProvenanceRef`, `AuditMetadata` and a typed `TraktError`. `to_dict()` is
   documented as excluding storage paths and stack traces. This *is* the
   agent-facing envelope; it does not need designing.

3. **The entitlement model (`trakt_core/entitlement.py`, `resource.py`,
   `organisation.py`, `principal.py`).** `organisation × resource × capability`,
   with `authorise_resource_access()` checking in a deliberate order (scope →
   parse → entitlements present → membership → capability → tenant → catalogue)
   so an ungranted resource and a nonexistent one are indistinguishable. A
   resource that the data cannot partition is refused at config load, not at
   request time. This is precisely the model a Buyer Agent and a Seller Agent
   need, and it is already written and tested.

4. **The deterministic calculation estate.** `mi_agent/concentration_tests/`
   with 24 registered evaluators (`metrics.py:952-977`) and an explicit design
   rule — *"There is no formula language and no path from client text to
   executable code: an unsupported test raises a structured implementation
   request, never an improvised calculation"* (`config/risk/concentration_test_library.yaml`).
   Plus `mi_agent/period_change/`, `analytics_lib/`, `mi_agent/mi_query_executor.py`.

5. **The OCC operational spine (`operations_control/`).** `WorkflowRun` with an
   enforced transition table, `GovernedAgentResult`, `DecisionRequired`,
   `OpsStore.append_audit` (hash-chained, with `verify_audit_chain`), and
   `InformationRequest` with `open → sent → answered → accepted/rejected` plus
   evidence references. The DD request/response object is a specialisation of
   something already in the codebase.

### The 3–5 biggest blockers

**B1 — There is no typed tool surface; there is one natural-language box.**
The only agent-reachable business capability is `mi.question.answer`, which takes
`question: str`. An external Claude or GPT agent calling it is passing prose to
Trakt's own LLM interpreter, which proposes an `MIQuerySpec` that a validator may
reject. That is two probabilistic hops where the architecture calls for one. The
~25 richer routes (`/mi/risk-limits`, `/mi/concentration-tests`,
`/mi/evolution/*`, `/mi/cohorts/*`, `/mi/geo/exposure`, `/mi/forecast/*`) are
FastAPI handlers in `app.py`, not governed capabilities: they do not return
`GovernedResult`, and the governance doc states plainly that "the risk and
forecast routes are not scope-gated at all yet". Exposing them to agents as they
stand would export an ungoverned surface.

**B2 — No machine identity, and the permission model is switched off.**
`config/` contains `tenancy.example.yaml`, `organisations.example.yaml`,
`resources.example.yaml`, `entitlements.example.yaml`, `principals.example.yaml`
— every one an example. `identity.py` documents this as "compatibility mode":
`resolve_organisation` returns `None`, `context.entitlements` stays `None`, and
the deployment authorises through `authorise_portfolio_access` against a tenancy
registry that also does not exist, leaving the honest limitation the doc states:
"the tenant owns any well-formed selector inside its own namespace". There is no
service-account registration, no client-credentials flow other than an Entra app
registration for Copilot, and no way to mint a scoped token for "Buyer Agent
acting for Firm X on portfolio Y".

**B3 — No loan-level or value-level read path.** Everything an agent can reach is
a portfolio aggregate. There is no `get_loan(loan_id)`, and critically no
`explain_value(loan_id, field)`. The raw material exists —
`lineage_tracker.py` emits `field_lineage.json` and `value_lineage.json`,
`engine/provenance.py` stamps six provenance fields on every row,
`delta_manifest.py` hashes inputs — but none of it is served at runtime. So Trakt
can currently tell an agent *"the balance is £183,450"* with a snapshot id and
approval state, but not *"…mapped from OUT_PRIN under alias set vX, validated
against BAL101, effective 31 July"*. For acquisition DD, the second sentence is
the product.

**B4 — No durable A2A workflow object and no outbound seam.** There is no
portfolio-scoped request/response object. `InformationRequest` is bound to an
`OnboardingCase` and its items are catalogue field references
(`{"section", "field", "index", "label"}`), not portfolio/loan scopes. And the
governance doc names the gap itself: *"No outbound notification seam — an
agent-to-agent workflow can call in, but Trakt cannot call back."*

**B5 — Two live parallel implementations of the same analytics.** Already
documented internally (`docs/spine_audit_single_source_of_truth.md`):
`analytics/mi_prep.py` vs `mi_agent_api/funded_prep.py` (different numeric
parser, bucketing, LTV/age derivation); `analytics/risk_monitor.py` vs
`mi_agent/risk_monitor/`; `pipeline_prep` in three places. Plus percent-scale
decided in Python and re-decided differently in React. A human UI can tolerate
two answers that differ in the last decimal. An agent that raises a DD request on
the basis of one of them cannot.

### Is a major architectural rewrite required?

**No.** Nothing found in this review calls for one. The governance core, the
canonical model, the calculation engine and the audit spine are all correctly
placed and correctly layered. Every recommendation below is an *extension*, a
*wrapper*, or a *configuration activation* of something already present. The one
structural debt worth paying down (B5) is a consolidation, not a redesign, and is
already scoped in an existing internal audit.

### How difficult is an external-agent prototype?

**Low–moderate. Roughly 2–4 engineering weeks.** The path is: define a
`trakt_tools` capability registry over 12–15 typed functions that wrap existing
implementations; add one `ExecutionContext` adapter for a service identity;
activate `config/tenancy.yaml` and `config/entitlements.yaml`; expose the tools
twice over one registry (REST/OpenAPI + a thin MCP server). No calculation is
written. The largest genuinely new pieces are `get_loan` and `explain_value`.

### How difficult is the synthetic Buyer/Seller demo?

**Moderate. Roughly 4–6 further weeks on top of Phase 1.** The new work is a
`DDRequest`/`DDResponse` document pair with a state machine (a near-copy of
`InformationRequest` re-scoped to portfolio/loan), a second entitled organisation
in configuration, and two thin agent harnesses. The portfolio itself is free:
`simulation/` already generates deterministic, seed-controlled multi-asset-class
portfolios with independently computed expected truth, and `demo_platform/`
already drives the real API against a synthetic pack.

---

## 2. Current architecture map

### 2.1 The shape

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ INTERFACES                                                               │
 │   React MI Agent      frontend/mi-agent-ui/          POST /mi/query       │
 │   React OCC           frontend/operations-control-ui/                     │
 │   M365 Copilot        deploy/copilot-agent/          POST /v1/copilot/…   │
 │   Teams bot           mi_agent_api/teams_bot.py      (notification only)  │
 │   Streamlit (legacy)  analytics/streamlit_app_erm.py (parallel path)      │
 │   CLIs                engine/*/cli.py, simulation/runner.py               │
 └────────────────────────────────┬─────────────────────────────────────────┘
                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ IDENTITY ADAPTERS            mi_agent_api/identity.py, auth.py,           │
 │                              copilot_auth.py                             │
 │   Easy Auth header ─┐                                                    │
 │   Entra bearer JWT ─┼──► ExecutionContext(tenant, actor, channel, scopes,│
 │   in-process       ─┘        organisation, entitlements, request_id)     │
 └────────────────────────────────┬─────────────────────────────────────────┘
                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ GOVERNED CAPABILITIES  (only two exist)                                  │
 │   mi.question.answer          mi_agent_api/mi_service.py                 │
 │   artefact.investor_pack.get  mi_agent_api/artefacts.py                  │
 │      scope → portfolio authorisation → source approval → execute         │
 │                                                                          │
 │ UNGOVERNED ROUTES (~25)  mi_agent_api/app.py — risk-limits, concentration│
 │   tests, evolution, cohorts, geo, forecast, snapshots, insights, decks    │
 └────────────────────────────────┬─────────────────────────────────────────┘
                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ DOMAIN                                                                   │
 │   mi_agent/mi_query_executor.py     deterministic aggregation            │
 │   mi_agent/concentration_tests/     covenant + limit engine (24 metrics) │
 │   mi_agent/period_change/           two-snapshot movement + bridge       │
 │   mi_agent/risk_monitor/            migration · concentration · flags    │
 │   analytics_lib/                    buckets · stratify · concentration   │
 │   mi_agent/interpreter/             NL → MIQuerySpec (LLM behind Protocol)│
 └────────────────────────────────┬─────────────────────────────────────────┘
                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ CANONICAL DATA + PIPELINE                                                │
 │   Gate 1 semantic_alignment.py     raw headers → canonical fields        │
 │   Gate 2 canonical_transform.py    typing · geo · LTV · derivations      │
 │   Gate 2.5 lineage_tracker.py      field_lineage.json · value_lineage    │
 │   Gate 3 validate_canonical.py + validate_business_rules.py              │
 │   Gate 4 regime_projector.py / annex12_projector.py                      │
 │   Gate 5 xml_builder_*.py + XSD validation                               │
 │   engine/provenance.py             source-cohort stamping (fail-closed)  │
 │   engine/platform_assembler.py     per-portfolio → central canonical     │
 │   snapshot/store.py                idempotent content-hashed snapshots   │
 └────────────────────────────────┬─────────────────────────────────────────┘
                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ GOVERNANCE CORE           trakt_core/                                    │
 │   context · tenancy · organisation · principal · resource · entitlement  │
 │   policy · runtime · errors · envelope · audit · portfolio               │
 └──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Traced flow 1 — raw tape → canonical / OCC

Entry: `engine/orchestrator/trakt_run.py`, or the governed conductor
`engine/orchestrator_agent/`.

1. **Gate 1 — semantic alignment.** `engine/gate_1_alignment/semantic_alignment.py`
   (702 lines) resolves raw headers to canonical fields through six deterministic
   tiers: exact, normalised, alias lookup, token-set Jaccard, RapidFuzz. Aliases
   live in `config/system/aliases_{mandatory,optional,analytics,onboarding_*}.yaml`
   and `engine/gate_1_alignment/aliases/`.
2. **Tier 7 (optional, human-gated).** `agent_orchestrator.py` → `llm_mapper_agent.py`
   calls Claude for headers still unmapped or below `review_threshold` (0.92),
   nulls hallucinated field names not in the registry, requires human confirmation
   (`auto_approve_threshold: null` by default), and persists confirmed mappings to
   `aliases_llm_confirmed.yaml` so the next run resolves at Tier 3 with no LLM. A
   versioned JSON session record is written to `governance/agent_sessions/`.
   *This is the single best existing example of the "AI reasons, Trakt decides"
   principle already implemented.*
3. **Transform.** `engine/gate_2_transform/canonical_transform.py` (1,155 lines)
   applies typing (`apply_types`), enum normalisation, geography enrichment
   (`normalize_geography`, NUTS/ITL lookup), and LTV resolution (`_resolve_ltv`,
   line 762) which brings supplied LTV onto percentage points, reconciles it
   against balance/valuation, and derives only what is absent — with an explicit
   disclosure flag (`GATE2_LTV_ACQUIRED_DISCLOSURE`, default off) governing
   whether LTV may be derived from an outstanding rather than principal balance.
   Declarative derivations come from `config/system/canonical_derivations.yaml`
   against a fixed rule library; the config states "*nothing here can invent a
   value: every rule maps null to null*".
4. **Provenance stamping.** `engine/provenance.py` stamps
   `source_portfolio_id`, `source_portfolio_type`, `source_portfolio_label`,
   `acquisition_date`, `seller_name`, `portfolio_cohort` on every row from
   run-level metadata, failing closed rather than assigning `unknown`.
5. **Lineage.** `engine/gate_2_transform/lineage_tracker.py` emits
   `field_lineage.json` always and `value_lineage.json` on request.
6. **Validation.** `validate_canonical.py` (schema/format) then
   `validate_business_rules.py` (889 lines, cross-field, rule ids `DAT001…`,
   `BAL101…`, `LTV001-004`, `ARR001-002`, `DEF001-002`, `SEC001`, `REC001`,
   `PROV001-005`), aggregated by `aggregate_validation_results.py`.
7. **Assembly.** `engine/platform_assembler.py` / `assembler_agent.py` selects the
   latest accepted snapshot per `source_portfolio_id`, rejects duplicate composite
   keys, and writes `platform_canonical_manifest.json` with `content_sha256` and
   per-portfolio `input_file_hash`.

**Assessment:** this is the strongest part of the system. Provenance is stamped,
lineage is emitted, hashes are recorded, and the fail-closed posture is real.

### 2.3 Traced flow 2 — canonical → validation/calculation → output

- **MI:** `mi_agent_api/datasets.py` resolves the active dataset (preferring
  `platform_canonical_typed.csv`), `funded_prep.prepare_funded_mi_dataset`
  prepares it, `ParsedQuestion.parse` produces a spec, `chat_routing.try_route`
  offers ~12 specialised routes, and anything unmatched falls to
  `mi_query_executor.execute_mi_query`. The executor emits a `reconciliation`
  block (`_build_reconciliation`, line 506) recording rows/balance before and
  after filtering — a coverage disclosure that already exists.
- **Covenants/limits:** `mi_agent/concentration_tests/evaluation.py::evaluate_active_tests`
  returns per-test current value, prior value, movement, status transition,
  utilisation, headroom, breach amount and configuration provenance;
  `forward.py` adds forward states, pipeline drivers and
  `expected_breach_horizon`.
- **Period change:** `mi_agent/period_change/workflow.py` resolves two snapshots,
  calculates metric changes, distribution changes and a balance bridge, and
  carries a `CALCULATION_VERSION`. Its docstring states the invariant well:
  *"the summary block is generated from the calculated tables; it cannot introduce
  a fact that is not already in metric_changes, distribution_changes or
  balance_bridge."*
- **Regulatory:** `regime_projector.py` → `annex2_delivery_normalizer` →
  `xml_builder_annex2.py` → XSD validation, wrapped by
  `engine/annex_delivery_agent/` which adds tenant binding, deterministic run
  identity, restart safety, an approval gate and disclosure of everything the
  builder filled in or coerced.

### 2.4 Traced flow 3 — user request → backend → result

`POST /mi/query` (`app.py:1722`) is a genuinely thin adapter: build context,
translate body, call `mi_service.execute_governed_mi_query`, present. The
capability runs scope check → `authorise_portfolio_access` → `evaluate_source_approval`
→ analysis, with the ordering comment *"nothing above touches data; every check
precedes the first read"*, and returns `GovernedResult` with an
`emit_audit_event` on every path including refusals.

The other ~25 routes do not follow this pattern.

### 2.5 Traced flow 4 — existing AI interaction

Two distinct patterns, both correct in principle:

- **Interpretation.** `mi_agent/interpreter/` — a deterministic baseline
  (`deterministic.py`), an Anthropic adapter behind `AnthropicMIInterpreterClient`
  (a `Protocol`), golden examples, and an evaluator. The LLM emits *spec JSON
  only*; every candidate is passed through `MIQuerySpec.normalized()` and
  `validate_query_spec()` before it can run, and an invalid or ambiguous
  interpretation forces a clarification rather than executing.
- **Operational reasoning.** `operations_control/occ_agent/service.py` is
  explicitly described as "the OCC Agent's typed tool surface… There is no
  general-purpose 'do what the text says' entry point, and the interpreter cannot
  reach the store, the filesystem or the pipeline except through these methods."
  Its execution order — check state permits → do deterministic work → derive state
  from what controls returned → assert transition legal → persist → audit — is
  exactly the discipline an A2A tool layer needs.

**No native LLM tool-calling / function-calling exists anywhere in the
repository.** A repo-wide search for `tool_use`, `tools=`, `tool_choice`,
`function_call` returns nothing. Every LLM interaction is prompt-in / JSON-out.

### 2.6 Traced flow 5 — authn/authz to data access

React: SWA/Easy Auth → `X-MS-CLIENT-PRINCIPAL` → `auth.parse_principal` →
`identity.context_from_principal` (which calls `require_trustworthy_platform_auth`
and refuses in production when running in Azure with `WEBSITE_AUTH_ENABLED` unset)
→ `ExecutionContext` with `DEFAULT_MI_SCOPES`.

Copilot: `copilot_auth.copilot_auth_guard` validates signature/issuer/audience/expiry
against an allow-list of Entra directories' JWKS (reading `tid` unverified *only*
to select which allow-listed key set to verify against) →
`identity.context_from_copilot_principal` → resolves principal binding, then
organisation, then entitlements, and freezes all three onto the context.

Then: `authorise_portfolio_access` returns an `AuthorisedPortfolio` **token**, and
dataset resolution takes the token rather than a raw string — so there is no code
path from a request field to a dataframe that skips the check. Selectors must match
`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`, so a value cannot traverse a storage path.

**The honest limitation, stated in the code's own documentation:** with no
`config/tenancy.yaml` present the deployment is single-tenant and rule 3 (the
portfolio allow-list) is inert.

---

## 3. Existing capability inventory

| Capability | Exists? | Maturity | Current implementation | Reusable for A2A? | Gap |
|---|---|---|---|---|---|
| Canonical schema | Yes | High | `config/system/fields_registry.yaml` — 499 fields; `layer` ∈ {core 295, performance 93, product 57, collateral 54}; `category` ∈ {regulatory 316, analytics 183}; `portfolio_type`; `regime_mapping` to ESMA Annex 2/3/4/8/9 codes; `core_canonical` flag | Yes, directly | No entity/relationship declaration; `unit` present on only some numeric fields |
| OCC / canonical instance | Yes | High | `*_canonical_typed.csv` → `engine/platform_assembler.py` → `platform_canonical_typed.csv` + manifest with `content_sha256` | Yes | File-based; no loan-level query service |
| Field registry | Yes | High | Above, plus `config/business_semantics_registry.yaml` (242 curated fields with `analytical_concept`, `analytical_role`, `temporality`, `directionality`, `default_aggregation`, `weight_field`, `portfolio_comparability`, `confidence`, `rationale`) and `mi_agent/mi_semantics_field_registry.yaml` | Yes — this is the agent-facing vocabulary | Three registries; relationship between them is generator-driven (`scripts/build_business_semantics_registry.py`), not queryable at runtime |
| Mappings | Yes | High | `semantic_alignment.py` Tiers 1-6 + `config/system/aliases_*.yaml` + `aliases_llm_confirmed.yaml`; `alias_builder.py` (TF-IDF) | Yes | Mapping decisions are pipeline-time artefacts; no runtime `resolve_field` service |
| Validation | Yes | High | `validate_canonical.py` (schema/format), `validate_business_rules.py` (889 lines, ~40+ rule ids), `aggregate_validation_results.py`, `exception_db.py` (SQLite, hash-chained remediations) | Yes | Not exposed as a capability; `exception_db` is a separate SQLite store not wired to the governed path |
| Calculations | Yes | High | `mi_query_executor.py`, `analytics_lib/{buckets,stratify,concentration,cohort}`, `concentration_tests/metrics.py` (24 evaluators), `period_change/`, `risk_monitor/`, `forecast_extrapolation.py`, `scenario_engine.py` | Yes — this is the core asset | Not individually addressable; reachable only via NL question or UI-shaped routes |
| Provenance | Partial | Medium | `engine/provenance.py` (row-level source cohort), `lineage_tracker.py` (`field_lineage.json`, `value_lineage.json`), `delta_manifest.py` (SHA256), `SnapshotRef` + `ProvenanceRef` in the envelope | Envelope: yes. Lineage: as data, not as a service | No runtime `explain_value`; `ProvenanceRef` today only references `sourceNotes` and the reconciliation footer |
| Audit | Partial | Medium | `trakt_core/audit.py` (one structured line per governed execution, with a `_FORBIDDEN_KEYS` denylist); `OpsStore.append_audit` (hash-chained, `verify_audit_chain`); `exception_db` remediation chain | Yes as a pattern | Governed audit is log-only, not queryable; hash-chained audit is OCC/client-scoped, not capability-scoped |
| APIs | Partial | Medium | `mi_agent_api/app.py` (~35 routes), `operations_control/api/app.py` (~50 routes), `occ_agent/api.py` (~35 routes), `copilot_actions.py` (3) | Partly | Two governed capabilities; the rest are UI-shaped data endpoints, not business capabilities |
| Authentication | Yes | Medium-High | Easy Auth header (React) + Entra bearer JWKS validation (Copilot); `require_trustworthy_platform_auth` fails closed | Yes for humans and for an Entra-registered app | No non-Entra machine identity; no API-key/service-account path; no token minting |
| Authorization | Yes (dormant) | High (code) / Nil (config) | `trakt_core/tenancy.authorise_portfolio_access`; `trakt_core/entitlement.authorise_resource_access`; `resource.ResourceCatalogue`; scopes `portfolio:read`, `mi:query`, `artefact:read`, `artefact:generate`, `risk:read`, `forecast:read` | Yes, directly | All configuration files are `*.example.yaml`; risk/forecast routes not scope-gated; per-field and per-document authorisation do not exist |
| Machine identity | Partial | Low | `ACTOR_SERVICE` constant; Copilot app-only tokens set `actor_type=ACTOR_SERVICE` when `scp` is absent | Constant is right | No registration, no credential issuance, no "agent acting on behalf of user" delegation, no per-agent scope narrowing |
| Agent tools | No | — | `occ_agent/service.py` is a *typed tool surface* for the OCC's own interpreter, in-process only | Pattern is reusable | No externally callable tool registry, no JSON Schema exposure, no per-tool permission binding |
| LLM abstraction | Partial | Medium | `mi_agent/interpreter/anthropic.py` behind `AnthropicMIInterpreterClient(Protocol)`; `engine/onboarding_agent/llm_policy.py` has a `provider` field defaulting to `"anthropic"` | Yes | Every concrete adapter is Anthropic; `provider` is read but only one branch exists (`llm_mapping_reviewer.py:300`) |
| Workflow state | Yes | High | `operations_control/contracts.py` (`WorkflowRun`, 9 statuses, enforced `RUN_TRANSITIONS`); `onboarding/case.py` (`OnboardingCase`, 9 statuses + `InformationRequest` 5 statuses); `occ_agent/states.py` (19 states) | Yes as a pattern; `InformationRequest` almost directly | All three are onboarding/delivery-scoped. Nothing portfolio- or DD-scoped. No cross-organisation workflow |
| Structured messaging | Partial | Medium | `GovernedAgentResult`, `DecisionRequired`, `InformationRequest`, `trakt_notifications/contract.py` (versioned envelope, deterministic ids) | Yes as a pattern | No inter-party message contract; no correlation across organisations |
| Event model | Partial | Low-Medium | Azure Event Grid blob trigger (`function_app.py`); `trakt_notifications/outbox.py` (durable at-least-once with idempotency); `operations_control/intake.py` (idempotency keys) | Outbox pattern is reusable | No general event bus; **no outbound seam from a governed capability** (named as a known gap in the governance doc) |
| Document access | Partial | Medium | `copilot_artifacts.py` registry (5 registered types: investor deck, canonical tape, mapping report, validation report, ESMA XML) + `artefacts.get_investor_pack` (governed) + HMAC-signed short-lived download tokens | Yes | Only one type goes through the governed capability; the rest resolve directly. No loan-level document store, no evidence-document linkage |
| Testing | Yes | High | 225 test modules; 11 governance-specific suites including a dependency-direction subprocess check; `simulation/` computes expected truth independently of production code; `demo_platform/` drives the real API | Yes — an excellent base for tool contract tests | No contract tests for an agent interface (none exists yet) |

---

## 4. Canonical / ontology assessment

### 4.1 What already exists

Trakt has **three layered registries**, and together they are considerably more
than a field list:

**Layer 1 — `config/system/fields_registry.yaml` (499 fields).** The canonical
field contract. Each entry carries `format`, `allowed_values` (referencing
controlled enums in `enum_mapping.yaml`), `portfolio_type`, `category`, `layer`,
`core_canonical`, sometimes `unit`, and `regime_mapping` giving the ESMA code and
priority per annex. Example:

```yaml
current_loan_to_value:
  format: decimal
  unit: percentage_points
  category: regulatory
  portfolio_type: common
  layer: collateral
  core_canonical: false
  regime_mapping:
    ESMA_Annex2: {code: RREC12, priority: Analytics}
    ESMA_Annex3: {code: CREL76, priority: Mandatory}
    ESMA_Annex9: {code: ESTC13, priority: Analytics}
```

**Layer 2 — `config/business_semantics_registry.yaml` (242 curated fields).** The
analytical meaning layer, generated by `scripts/build_business_semantics_registry.py`
from a curated allowlist. It carries a genuine controlled taxonomy: 20
`analytical_concepts` (cashflow, collateral, coverage, credit_quality,
data_quality, eligibility, exposure, forecast, geography, leverage, liquidity,
loss, maturity, operational_performance, origination, payment_performance,
pricing, product_mix, tail_risk, valuation), 28 `categories`, four
`analytical_roles`, four `temporality` values, five `default_aggregations`,
`directionality`, `portfolio_comparability` and `confidence`. Example:

```yaml
current_loan_to_value:
  analytical_concept: leverage
  analytical_role: measure
  temporality: point_in_time
  directionality: higher_is_worse
  default_aggregation: weighted_average
  weight_field: current_outstanding_balance
  portfolio_comparability: comparable
  supports_materiality_assessment: true
  asset_applicability: [cross_asset]
  confidence: high
```

**Layer 3 — `mi_agent/mi_semantics_field_registry.yaml`.** The query-facing
semantics the executor resolves against (`role`, `allowed_aggregations`,
`allowed_chart_roles`, `synonyms`, `bucket_field`, `mi_tier`), projected to
clients by `mi_agent_api/catalogue.py`.

### 4.2 Does the Capability A example already work?

**Yes, for the mapping direction, and it is deterministic.** `OUT_PRIN`,
`Current Balance` and `BAL_01` all resolve to `current_outstanding_balance`
through Tiers 1-6 of `semantic_alignment.py` given aliases, and where they do not,
Tier 7 proposes with mandatory human confirmation and *learns* the alias so the
next run is deterministic. An external agent already never needs to know a
customer column name — it asks in canonical terms.

**What is missing is the reverse direction at runtime.** An agent cannot ask
"which source field produced this canonical value, under which alias set?" There
is no service; there is a JSON artefact.

### 4.3 Is the field registry an embryonic ontology?

**Yes — it is roughly an ontology's *attribute* layer with an implicit entity
layer and no relationship layer.**

- **Entities are implicit in `layer`.** `core` (295), `performance` (93),
  `product` (57), `collateral` (54). These are groupings, not entity types with
  identity.
- **Entity keys exist as ordinary fields.** `loan_identifier` (`core_canonical: true`),
  `borrower_identifier` (`portfolio_type: sme`, `core_canonical: true`),
  `original_obligor_identifier`, plus `snapshot/model.py` reserved columns
  (`snapshot_id`, `client_id`, `loan_id`) and the optional `spv_id`. Nothing
  declares that `borrower_identifier` is a *key of a Borrower* or that a Borrower
  has many Loans.
- **Relationships are encoded in code, not data.** `concentration_tests/metrics.py`
  knows that grouping by `borrower_id` gives borrower aggregation
  (`_eval_multi_loan_borrower_share`, `_eval_borrower_aggregate_share`) — that is
  a Loan→Borrower relationship expressed as a Python evaluator. Similarly the
  `field_roles` block in `config/risk/concentration_test_library.yaml` declares
  role → candidate-column resolution, which is a *partial* relationship
  declaration already in config.
- **Some target concepts are present as fields**, contrary to what an older
  internal audit records: `ifrs9_stage`, `ifrs9_stage_current`,
  `ifrs9_stage_previous`, `internal_risk_stage`, `indexed_loan_to_value`,
  `indexed_value`, `exposure_at_default`, `loss_given_default`,
  `number_of_days_in_arrears`, `number_of_days_in_principal_arrears`,
  `covenant_breach_trigger`, `consequence_for_breach_of_financial_covenant`,
  `current_valuation_amount`/`_date`/`_basis`/`_method`, `cumulative_recoveries`,
  `expected_timing_of_recoveries`, `free_cashflow`. The registry is richer than
  `docs/mi_analytics_architecture_current_state_audit.md` (dated 2026-06-18)
  states; that gap has been partly closed since.

### 4.4 Which target concepts are explicitly represented?

| Concept | Represented as | Machine-readable relationship? |
|---|---|---|
| Loan | Row + `loan_identifier`; `layer: core` | Implicit (row identity) |
| Borrower | `borrower_identifier`, `original_obligor_identifier`, borrower age/income fields | **No** — Loan→Borrower is code-only |
| Collateral | `layer: collateral` (54 fields), postcode/geography | **No** — 1:1 with loan is assumed |
| Valuation | `current_valuation_amount/_date/_basis/_method`, `original_valuation_amount`, `indexed_value` | Partial — dated but not a repeatable entity |
| Contract | Product/rate/term fields, `layer: product` | **No** |
| Cashflow | `free_cashflow`, `accrued_interest_in_period`, `cumulative_prepayments`, `cumulative_recoveries`, `net_periodic_payment_made_by_swap_provider` | **No** — period flows, no cashflow schedule entity |
| Performance state | `account_status`, `ifrs9_stage*`, `internal_risk_stage`, `arrears_balance`, `number_of_days_in_arrears` | Partial — `mi_agent/states/` and `config/mi/state_library.yaml` give a state vocabulary |
| Servicer | `seller_name` (provenance), agent-bank LEI fields (CRE) | **No** |
| Facility | `KIND_FACILITY` in `trakt_core/resource.py` | **Declared as a resource kind, no data model** |
| Covenant | `config/risk/concentration_test_library.yaml` + client-approved thresholds via `concentration_tests/store` | **Yes, best-modelled concept in the system** |
| Document | `copilot_artifacts` registry (5 types); `InformationRequest.evidence` refs | Partial — artefact-level, not loan-level |
| Event | Snapshot registration, audit records, workflow transitions | Partial — no unified event entity |

### 4.5 Is a graph database necessary?

**No.** Nothing found justifies it, and adopting one would be the single most
expensive way to solve a problem the codebase does not have. The reasons:

- The dominant access pattern is *set-based aggregation over a wide typed frame*
  (sum balance by region, weighted-average LTV by vintage, share-of-balance
  against a threshold). Pandas over a canonical CSV/parquet frame is the right
  engine for that; a graph is the wrong one.
- The relationship cardinality in scope is shallow: Loan→Borrower,
  Loan→Collateral, Loan→Valuation(dated), Loan→SourcePortfolio, Portfolio→SPV.
  Two or three hops, expressible as declared foreign keys.
- Every existing control — validation rules, entitlement predicates
  (`ResolvedResource.predicate()`), attribution checks — is written against a
  columnar frame. Migrating would invalidate them all.
- The "ontology" the requirement actually describes is a *shared vocabulary with
  declared semantics*, which Trakt already has in two of three layers.

### 4.6 Recommended incremental extension

**Extend the existing YAML registries. Do not add a layer, a database or a
triple store.**

**E1 (NOW) — declare entities and keys in the field registry.** Add two optional
keys to each field entry:

```yaml
loan_identifier:
  entity: loan
  entity_role: key          # key | attribute | foreign_key
borrower_identifier:
  entity: loan
  entity_role: foreign_key
  references: borrower.borrower_id
current_valuation_amount:
  entity: valuation
  entity_role: attribute
```

Plus one small new file, `config/system/entity_model.yaml`, declaring ~8 entities
(loan, borrower, collateral, valuation, contract, cashflow, source_portfolio,
document) with their key field, their parent, and their cardinality. That file is
what a tool like `describe_entity_model()` returns to an agent, and what
`get_loan()` uses to assemble a nested loan object from a flat row.

Both additions are purely additive: nothing reads the new keys until a tool does,
and the existing 499 entries stay valid.

**E2 (NEXT) — promote `field_roles` to a first-class concept resolver.**
`config/risk/concentration_test_library.yaml` already declares logical roles
(`balance_current`, `region`, `valuation_indexed`, `borrower_id`, …) resolved to
"the first present-and-populated canonical column". That is exactly the
customer-independent indirection Capability A asks for, and it is currently only
available to the concentration engine. Lift it into
`config/system/field_roles.yaml` and let every tool resolve through it. This is
the cheapest single step toward "an external agent never needs to know a column
name".

**E3 (LATER) — versioned concept identifiers.** Give each business-semantics
concept a stable id and version (`concept: leverage.current_ltv@2`) so a tool
response can cite which definition produced a number. Only needed when two
clients disagree about a definition.

---

## 5. Deterministic tool inventory

Sixteen tools sufficient for a first agentic portfolio review or acquisition-DD
workflow. Every one is deterministic; none requires an LLM. **Twelve wrap
existing implementations; four are genuinely new.**

Common contract for all: input includes `portfolio_id` (or `resource_ref`) and
optional `as_of_date`; output is `GovernedResult.to_dict()`, so every response
already carries capability id, schema version, status, request id, correlation
id, tenant, snapshot, policy state, scope coverage and typed error.

| # | Tool | Purpose | Existing implementation | Gap | Inputs | Outputs | Evidence returned |
|---|---|---|---|---|---|---|---|
| 1 | `list_portfolios` | Enumerate portfolios/resources this caller may reach | `trakt_core.entitlement.permitted_resources_for`, `mi_agent_api/portfolio_context.resolve_context`, `snapshots.py` | Thin wrapper | — | `[{resource_ref, kind, label, snapshot_id, reporting_date, row_count}]` | Snapshot id, content hash, approval state |
| 2 | `describe_dataset` | Identity + approval state of the governed dataset | `datasets.describe_active_dataset` → `_snapshot_ref` (`mi_service.py:181`) | Thin wrapper | `portfolio_id` | `SnapshotRef` + `PolicyState` | The whole response is evidence |
| 3 | `describe_field_catalogue` | The canonical vocabulary an agent must speak | `mi_agent_api/catalogue.build_catalogue`, `config/business_semantics_registry.yaml` | Thin wrapper + merge business semantics | `concept?`, `role?` | `[{canonical_field, display_name, analytical_concept, role, temporality, unit, default_aggregation, weight_field, directionality}]` | Registry version, `metadata.version` |
| 4 | `resolve_source_field` | Customer column → canonical concept, and back | `engine/gate_1_alignment/semantic_alignment.py` Tiers 1-6 | **Moderate** — needs a runtime service wrapping the tier ladder without running a pipeline | `source_header`, `portfolio_type` | `{canonical_field, tier, confidence, alias_source}` | Which alias file and tier resolved it |
| 5 | `portfolio_summary` | Loan count, balance, WA LTV, WA rate, key stats | `mi_query_executor._execute_summary`; `analytics_lib.numeric` | Thin wrapper | `portfolio_id`, `filters?`, `lens?` | `{loan_count, total_balance, currency, metrics[]}` | Reconciliation block (rows/balance before/after filter) |
| 6 | `stratify` | Distribution of a measure by a dimension or bucket | `analytics_lib/stratify.py`, `analytics_lib/buckets.py`, `_execute_grouped` | Thin wrapper | `dimension`, `measure`, `aggregation?`, `bucket_config?`, `filters?` | `{rows[{label, value, share, count}], total, excluded}` | Bucket config key + version, excluded/missing counts |
| 7 | `concentration` | Top-N and group shares against a dimension | `analytics_lib/concentration.{group_shares,top_n_concentration,limit_usage,rag_status}` | Thin wrapper | `dimension`, `n?`, `basis?` | `{top_n_share, groups[], hhi?}` | Denominator basis, rows contributing |
| 8 | `evaluate_covenants` | Run the client's activated concentration/covenant tests | `mi_agent/concentration_tests/evaluation.evaluate_active_tests` + `summarise` | Thin wrapper; needs `risk:read` scope gate turned on | `portfolio_id`, `as_of?`, `prior_as_of?`, `category?` | `{tests[{test_id, metric_id, value, threshold, operator, status, utilisation, headroom, breach_amount, movement, prior_status}], summary}` | Configuration version + approval provenance from `ActiveConfiguration`; per-test resolved column names |
| 9 | `covenant_drillthrough` | The loans behind one test result | `concentration_tests/evaluation.drillthrough` | Thin wrapper | `test_id`, `limit?` | `{loans[], contribution[]}` | Resolved role→column map, filter predicate |
| 10 | `risk_limit_status` | Schedule-8 style limit status incl. movement | `mi_agent_api/risk_limits.py` | Thin wrapper; route currently ungoverned | `portfolio_id`, `category?` | `{limits[{name, actual, limit, headroom, status, source, confidence, notes, missing_fields}]}` | Limit source (extracted config + file), confidence, missing fields |
| 11 | `period_change` | What changed between two snapshots, and why | `mi_agent/period_change/workflow.py` (+ `bridge.py`) | Thin wrapper | `baseline_date`, `current_date`, `fields?` | `{metric_changes[], distribution_changes[], balance_bridge, summary}` | `CALCULATION_VERSION`, both snapshot ids, audit record |
| 12 | `rank_loans` | Top/bottom exposures by a measure | `mi_query_executor._execute_ranked_loans` | Thin wrapper | `sort_by`, `direction?`, `limit`, `filters?` | `{rows[], total_rows}` | Filter predicate, excluded counts |
| 13 | `list_validation_exceptions` | Open data-quality and business-rule findings | `validate_business_rules.RULES`, `aggregate_validation_results.py`, `exception_db.py` | **Moderate** — needs a governed read over run artefacts / `exception_db` | `portfolio_id`, `severity?`, `rule_id?` | `{exceptions[{rule_id, severity, field, affected_rows, message, status}], summary}` | Rule id and rule text, validation run id |
| 14 | `data_completeness` | Field-level population and DD scorecard | `engine/onboarding_agent/target_coverage.py`, `domain_coverage.py`; `config/mna/diligence_scorecard.yaml` (declared, unread) | **Moderate** — the scorecard config exists but nothing reads it | `portfolio_id`, `field_set?` | `{fields[{canonical_field, populated, populated_pct, rag}], overall_rag}` | Threshold config version, denominator |
| 15 | `get_loan` | One loan as a structured, provenanced object | — | **NEW** — needs the entity model (E1) plus a row-level read path | `loan_id` | `{loan{}, borrower{}, collateral{}, valuation{}, performance{}}` | Per-field `source_field`, `effective_date`, `validation_status` |
| 16 | `explain_value` | Full provenance envelope for one value | `lineage_tracker.py` artefacts + `provenance.py` + validation results | **NEW as a service** — the data exists as JSON artefacts | `loan_id`, `canonical_field` | The minimum provenance envelope (§8) | The whole response is evidence |
| 17 | `list_documents` / `get_document` | Retrieve an evidence artefact | `copilot_artifacts` registry (5 types), `artefacts.get_investor_pack` (governed) | **Moderate** — generalise the governed path to all registered types | `artefact_type`, `portfolio_id` | `{artefact_type, label, reporting_period, download_url (signed, short-lived), content_hash}` | Content hash, reporting period, generating run id |
| 18 | `raise_dd_request` / `respond_dd_request` / `list_dd_items` | The A2A workflow verbs | `operations_control/onboarding/case.InformationRequest` (5 statuses, evidence refs) | **NEW, but as a re-scoping** — copy the object, change the scope from case+catalogue-field to portfolio+loan+field | See §9 | See §9 | Full state history + evidence refs |

**Explicitly not recommended as separate tools**, because they duplicate
existing determinism or would tempt an agent to calculate:

- No `calculate_ltv(loan_id)` as a standalone. LTV is already resolved once in
  `canonical_transform._resolve_ltv` under an explicit disclosure policy and is
  a *canonical field*. An agent should read `current_loan_to_value` through
  `get_loan`/`explain_value` and get the resolution method as provenance — not
  ask Trakt to recompute it a second way. Adding a recompute path is exactly the
  duplication risk the review is meant to catch.
- No `calculate_arrears`. `arrears_balance`, `number_of_days_in_arrears`,
  `number_of_days_in_principal_arrears`, `date_last_in_arrears` are canonical
  fields with ESMA codes. Aggregation is `stratify`/`portfolio_summary`.
- No `indexed_ltv` tool yet. `indexed_loan_to_value` and `indexed_value` are
  registry fields and `valuation_indexed` is a declared field role, but the
  concentration library marks index-dependent metrics `interface_only` —
  *"reports `external_reference_unconfigured` until the required external source
  is configured. Never simulated."* Expose it only once an approved HPI feed is
  configured; until then the honest tool response is the existing refusal.
- No `eligibility` tool in Phase 1. `eligibility` is a declared
  `analytical_concept` with no evaluator behind it. Building one is a genuine
  new business capability, not an exposure task, and belongs after the demo.

**Deterministic vs probabilistic:** all 18 are deterministic. The only
probabilistic element in the whole loop is the *external agent's choice of which
tool to call and with what arguments* — which is exactly where it should be.

**Permission requirements** map onto the existing scope vocabulary with no new
verbs: tools 1-3 → `portfolio:read`; 5-7, 11-12 → `mi:query`; 8-10 →
`risk:read`; 13-14 → `mi:query` (or a new `quality:read` if a data-quality-only
grant is wanted); 15-16 → a new `loan:read` (loan-level data is materially wider
than aggregate MI and must be separately grantable); 17 → `artefact:read`;
18 → new `dd:request` / `dd:respond`.

---

## 6. External agent interface assessment

### 6.1 What exists

| Option | State today | Fit for an external agent |
|---|---|---|
| **`POST /mi/query`** (React) | Governed capability; Easy Auth header identity | Poor. Header identity is only trustworthy behind SWA; takes prose. |
| **`POST /v1/copilot/mi/query`** (Copilot) | Governed capability; Entra bearer + JWKS validation; hand-authored OpenAPI 3.0.3 with 3 operations; signed download tokens | **Closest thing that works today.** But it is packaged for M365 Copilot, exposes one NL question and one artifact getter, and its response shape is Copilot-specific (`copilot_text.normalise_payload` for the Copilot renderer). |
| **FastAPI auto-generated OpenAPI** | Exists implicitly (3.1) | Explicitly rejected in the Copilot spec's own header: it "covers the whole internal API, which must not be exposed to the plugin". Correct — it would expose ~35 UI-shaped routes with no scope gating. |
| **MCP** | Does not exist | — |
| **Direct Python import** | `execute_governed_mi_query(request, context, deps)` | Works for in-process agents only. |

### 6.2 Are business capabilities exposed, or only data?

**Mostly data, shaped for one UI.** `GET /mi/evolution/funded`,
`/mi/insight/movement-detail`, `/mi/cohorts/vintages`,
`/mi/concentration-tests/drivers` are React panel feeds. They are close to
business capabilities semantically, but they return React-shaped payloads without
a `GovernedResult` envelope, without scope gating, and with camelCase keys chosen
for a specific frontend. Two things are genuinely business capabilities:
`mi.question.answer` and `artefact.investor_pack.get`.

### 6.3 Is MCP appropriate? — assessed, not assumed

**Arguments for MCP here:**
- Trakt's target consumers are named as Claude, OpenAI-based agents and Copilot.
  MCP is becoming the common denominator for tool exposure across those, and a
  client's own agent framework increasingly speaks it.
- MCP's model (a server advertising typed tools with JSON Schema) is a direct
  match for what §5 proposes, and the governance doc already anticipates it:
  *"MCP adapter — same three steps per tool. Return `result.to_dict()`… Gate each
  tool on a scope."*
- Tool *discovery* matters for a DD workflow where the agent decides which of 16
  tools to call. MCP gives that for free; a REST/OpenAPI client has to be told.

**Arguments against MCP as the *only* interface:**
- MCP's authentication and multi-tenancy story is weaker and less settled than
  bearer-token REST. Trakt's whole security posture rests on a signature-verified
  directory identity resolving an organisation and its entitlements. That maps
  cleanly onto OAuth2/Entra over HTTPS; it maps less cleanly onto some MCP
  transports.
- A client's proprietary agent, a partner's back-office system, or a CI job may
  not speak MCP at all. REST/OpenAPI is universal and Trakt already ships a
  hand-authored 3.0.3 spec, so the pattern is proven in-house.
- MCP servers are typically deployed per-connection; Trakt's governance requires
  a single audited server-side execution path.

**Recommendation: both, over one registry — REST/OpenAPI first.**

The decisive point is that this is not a real choice if the tool registry is built
correctly. Write the tools once as governed capabilities in Python. Then:

- **`trakt_agent_api` (REST/OpenAPI 3.0.3)** — one route per tool under
  `/v1/agent/tools/{tool_name}`, plus `GET /v1/agent/tools` returning the JSON
  Schemas. Entra client-credentials bearer auth, reusing `copilot_auth` as the
  pattern. **This is the minimum viable interface and the only thing Phase 1
  needs.**
- **`trakt_mcp` (thin MCP server)** — generated from the same registry,
  translating each entry into an MCP tool declaration and each call into the same
  `execute_governed_tool(...)`. Because it holds no logic, it is a few hundred
  lines and can lag or lead the REST surface without risk.

Tool schemas should be exposed as **JSON Schema generated from the tool
registry**, not hand-maintained. Given Python dataclasses or Pydantic models per
tool, both the OpenAPI document and the MCP tool list are derived artefacts, and
a test asserts they agree — the same discipline
`tests/mi_agent_api/tests/test_copilot_package.py` already applies to the Copilot
spec.

**Model-agnosticism holds by construction**, because the tools accept typed
arguments and return `GovernedResult`. Nothing in the tool layer knows what model
is calling. Trakt's own LLM usage (the Tier 7 mapper, the MI interpreter) is
orthogonal and stays internal.

### 6.4 Minimum architecture

```
  Claude / GPT / Copilot / client agent
              │  HTTPS + Bearer (Entra client credentials)
              ▼
  ┌──────────────────────────────────────────────────────────┐
  │ mi_agent_api/agent_api.py        (NEW — thin, ~300 lines)│
  │   GET  /v1/agent/tools           JSON Schema per tool    │
  │   POST /v1/agent/tools/{name}    invoke                  │
  │   auth: agent_auth.py (reuse copilot_auth pattern)       │
  │   → ExecutionContext(actor_type=ACTOR_SERVICE,           │
  │                      channel=CHANNEL_ENTERPRISE_AGENT)   │
  └───────────────────────────┬──────────────────────────────┘
                              ▼
  ┌──────────────────────────────────────────────────────────┐
  │ trakt_tools/registry.py          (NEW — the ONE registry)│
  │   ToolSpec(name, description, input_model, output_model,  │
  │            required_scope, handler, version)             │
  │   execute_governed_tool(name, args, context)             │
  │     → scope check → resource authorisation → handler      │
  │     → GovernedResult → emit_audit_event                  │
  └───────────────────────────┬──────────────────────────────┘
                              ▼
  ┌──────────────────────────────────────────────────────────┐
  │ EXISTING, UNCHANGED                                       │
  │   mi_query_executor · concentration_tests · period_change │
  │   analytics_lib · risk_limits · artefacts · datasets      │
  └──────────────────────────────────────────────────────────┘

  trakt_mcp/server.py (NEXT) reads the same registry — no second path.
```

Total genuinely new code for Phase 1: the registry, the auth adapter, the route
module, 12 thin handlers, and 2 new capabilities (`get_loan`, `explain_value`).

---

## 7. Identity and permissions assessment

### 7.1 What is required for each party

| Party | Needs | Exists today | Gap |
|---|---|---|---|
| **Human user** | Interactive sign-in, tenant from deployment config, MI scopes | Yes — Easy Auth (React) and Entra delegated tokens (Copilot). `PrincipalRegistry` gives per-individual kill switch. | None for Phase 1 |
| **Agent acting for a user** | Delegated token, user's identity preserved in audit, scopes ⊆ user's | Partial — a Copilot delegated token sets `actor_type=ACTOR_USER` when `scp` is present | No `on_behalf_of` field on `ExecutionContext`; audit records one actor, not (agent, principal) |
| **Customer's service agent** | Client-credentials identity, organisation resolved from directory, entitlements per resource | Model exists (`ACTOR_SERVICE`, `CHANNEL_ENTERPRISE_AGENT`, `resolve_entitlements`) | No registration path, no adapter, config dormant |
| **Synthetic Buyer Agent** | A distinct organisation with grants over the *seller's* portfolio limited to aggregate + DD verbs | `OrganisationRecord` with `ORG_TYPE_INVESTOR` exists; `Grant(organisation, resource, capabilities)` exists | Config files are examples; no `dd:*` capabilities; no loan-level scope |
| **Synthetic Seller Agent** | A distinct organisation with wider grants over its own portfolio, including loan-level and document access | Same model | Same, plus `loan:read` and `document:read` do not exist |
| **Future external institutional agent** | Everything above plus contract, rate limiting, key rotation, per-agent audit export | — | Deliberately LATER |

### 7.2 Can the current model support this?

**Yes — the model, not the deployment.** `trakt_core/entitlement.py` already
answers the exact question A2A poses: *may this organisation perform this
capability against this resource?* Its design decisions are correct for
institutional credit and would be expensive to retrofit:

- Explicit grants, never inferred roles: `organisation_type: warehouse_funder`
  grants nothing.
- Capabilities held *per resource*, so Buyer Agent holding `mi:query` on
  Portfolio A and nothing on Portfolio B is a natural expression.
- A resource the data cannot partition cannot be granted to anyone —
  `RESOURCE_NOT_PARTITIONABLE` is raised at config load.
- `resolve_entitlements` runs once, at context construction, and is frozen — no
  capability can widen its own authority mid-execution.
- Membership is checked before catalogue lookup, so an ungranted and a
  nonexistent resource are indistinguishable.
- `Grant.status ∈ {active, proposed, revoked}` with `approved_by`/`approved_at`,
  and the OCC Agent may only ever write `proposed` — *"only a human setting
  `status: active` grants anything"*.

### 7.3 Do we need enterprise ABAC?

**No.** The existing two-axis model — `ExecutionContext.scopes` (what this caller
may do at all) intersected with grants (what this organisation may do to this
resource) — is sufficient through Phase 3 and probably Phase 4. `ResolvedResource.predicate()`
already gives row-level attribute filtering where a resource is a partial book.
Adding a policy engine would be premature and would duplicate a control that is
already tested.

### 7.4 Incremental design

**P1 (NOW) — activate what exists.** Create real `config/tenancy.yaml`,
`config/organisations.yaml`, `config/resources.yaml`, `config/entitlements.yaml`,
`config/principals.yaml` from the `.example` files. This turns on portfolio
allow-listing (tenancy rule 3), organisation identity, and per-resource grants
with **zero code change**. It is the highest-value/lowest-cost item in this
review.

**P2 (NOW) — a service-identity adapter.** `mi_agent_api/agent_auth.py` +
`identity.context_from_service_principal(...)` producing
`ExecutionContext(actor_type=ACTOR_SERVICE, channel=CHANNEL_ENTERPRISE_AGENT,
scopes=<narrowed>, organisation_id=…, entitlements=…)`. Reuse `copilot_auth`
verbatim for token validation; the only new logic is narrowing scopes per
registered agent rather than granting `DEFAULT_MI_SCOPES`.

**P3 (NOW) — narrow scopes per agent.** Add an optional `scopes:` list to the
organisation record so a registered agent gets `{portfolio:read, mi:query,
risk:read}` and not `artefact:generate`. Today `_scopes_for()` returns
`DEFAULT_MI_SCOPES` for every role — fine for two human roles, wrong for agents.

**P4 (NEXT) — two new capabilities and delegation.** Add `loan:read` and
`document:read` to `KNOWN_CAPABILITIES` (loan-level data is materially wider than
aggregate MI and must be separately grantable), and add
`on_behalf_of: Optional[str]` to `ExecutionContext` with a matching field in
`AuditMetadata`, so "Buyer Agent acting for Jane at Firm X" is one audit line.

**P5 (NEXT) — `dd:request` / `dd:respond` capabilities**, granted asymmetrically:
Buyer holds `dd:request` on the seller's portfolio resource; Seller holds
`dd:respond`. That asymmetry *is* the demo's permission story, and it needs no
new mechanism.

**P6 (LATER) — field-level authorisation.** Only if a real counterparty requires
"aggregates yes, borrower names no". The mechanism would be a field allow-list on
the grant, filtered in the tool layer. Not needed for the demo.

---

## 8. Provenance assessment

### 8.1 What exists

| Element | Exists | Where | Served to a caller? |
|---|---|---|---|
| Source tracking (which book) | **Yes** | `engine/provenance.py` — 6 fields stamped on every row, fail-closed | Yes, via `SnapshotRef.source_portfolios` |
| Input file lineage | **Yes** | `platform_canonical_manifest.json` (`input_file_hash` per portfolio), `delta_manifest.py` SHA256 | No |
| Mapping lineage | **Yes, as artefact** | `field_lineage.json` from `lineage_tracker.py`; alias files; `governance/agent_sessions/` for LLM-assisted mappings | No |
| Value lineage | **Yes, optional artefact** | `value_lineage.json` | No |
| Timestamps | Yes | `AuditMetadata.started_at`, snapshot `created_at`, workflow `updated_at` | Yes (audit) |
| Effective / reporting date | Yes | `SnapshotRef.reporting_date`; `snapshot/model.py` enforces date separation (`upload_timestamp` must never become `reporting_date`) | Yes |
| As-of date | Partial | `MiQueryRequest.as_of_date`; `reporting_date` | Yes |
| Calculation methodology | Partial | `period_change.CALCULATION_VERSION`; `concentration_test_library` `schema_version`/`library_version`; derivation `rule_id`s in `canonical_transform` (e.g. `RESOLVE_CURRENT_LOAN_TO_VALUE`) | No — not in the envelope |
| Validation state | **Yes** | `validate_canonical` + `validate_business_rules` rule ids; `exception_db` findings | No |
| Source documents | Partial | `copilot_artifacts` registry; `InformationRequest.evidence` refs | Partly (artefacts) |
| Confidence indicators | **Yes** | Mapping tier confidence; `business_semantics_registry.confidence`; `risk_limits` per-limit `confidence` | Partly |
| Data overrides | **Yes** | `exception_db.remediations` — `original_value`, `override_value`, `justification`, `user_id`, hash-chained | No |
| Previous values | **Yes** | `period_change` prior snapshot; `ifrs9_stage_previous`; `bank_internal_rating_prior_to_default` | Via period_change only |
| User changes | **Yes** | `OpsStore.append_audit` (hash-chained, `verify_audit_chain`); `DecisionRequired` resolutions | No |
| Audit records | **Yes** | `trakt_core/audit.py` + `OpsStore` | Log only |

**The pattern is consistent and worth stating plainly: Trakt records almost
everything an agent needs and serves almost none of it.** The `ProvenanceRef`
docstring is explicit about the deliberate limit — *"Provenance is a reference,
not a second model… Row-level provenance stays where it is — in
`engine.provenance` and the canonical data itself."* That decision was right for
a UI. It is the binding constraint for an agent.

### 8.2 The minimum provenance envelope

Not every field above. The minimum for an agent to distinguish a number from a
fact:

```json
{
  "value": 183450.00,
  "unit": "GBP",
  "canonical_field": "current_outstanding_balance",
  "concept": "exposure",
  "entity": {"kind": "loan", "id": "LN-0001842"},

  "source": {
    "dataset_label": "ere_202607_central_tape.csv",
    "snapshot_id": "snap_7f3a91",
    "content_hash": "sha256:9c1e…",
    "source_portfolio_id": "acquired_001",
    "delivered_by": "Seller A"
  },
  "source_field": "OUT_PRIN",

  "effective_date": "2026-07-31",
  "as_of_date": "2026-08-11",

  "transformation": {
    "mapping_tier": 3,
    "mapping_source": "aliases_llm_confirmed.yaml",
    "mapping_version": "2026-06-14",
    "derivation_rule": null
  },

  "validation": {
    "status": "passed",
    "rules_applied": ["BAL001", "BAL101"],
    "exceptions": []
  },

  "calculation": {
    "method": "as_supplied",
    "method_version": null
  },

  "evidence": [
    {"kind": "validation_report", "ref": "artefact://ERE/2026-07/validation_report.json"},
    {"kind": "mapping_report",    "ref": "artefact://ERE/2026-07/mapping_report.json"}
  ]
}
```

For a **calculated** value (a covenant utilisation, a WA LTV, a period change),
the same envelope with `calculation.method` naming the registered evaluator and
`method_version` its library version, and `source_field` replaced by
`inputs: [{canonical_field, aggregation, weight_field, row_count}]`.

Nine top-level keys. Everything in it already exists somewhere in the codebase;
none of it is new metadata.

### 8.3 How it attaches without duplicating the data model

**Attach it to the *response*, not to the data.** Three rules:

1. **Aggregate answers keep the existing envelope.** `GovernedResult.provenance`
   already carries `SnapshotRef` + `sourceNotes` + reconciliation. Extend
   `ProvenanceRef` with two optional fields — `calculation: {method, version}` and
   `evidence: [{kind, ref}]` — and nothing else. Additive, no consumer breaks.

2. **Value-level provenance is produced on demand by `explain_value`, never
   stored per cell.** The inputs are all already persisted: `field_lineage.json`
   (mapping), the row itself (value + provenance fields), the validation results
   (rule outcomes), the snapshot manifest (hashes and dates). `explain_value`
   *joins* these at request time. This is the key design choice: it makes the
   envelope free at rest and costs one lookup only when an agent actually asks
   "why".

3. **`get_loan` returns values with a *compact* provenance stub per field**
   (`{source_field, effective_date, validation_status}`) and a `provenance_ref`
   the agent can expand via `explain_value`. Full envelopes for 60 fields on
   every loan read would be unusable; a stub plus a drill-down is what an agent
   actually needs.

**What not to add now:** confidence scores on every value, per-cell override
history in the envelope, or a provenance graph. `exception_db` already holds
override history with a hash chain; surface it through
`list_validation_exceptions` and let `explain_value` link to it.

---

## 9. Workflow / state assessment

### 9.1 What exists

Three real state machines, all with enforced transition tables:

1. **`WorkflowRun`** (`operations_control/contracts.py`) — 9 statuses, explicit
   `RUN_TRANSITIONS` dict, `IllegalTransition` raised by `transition()`, 10
   stages, `idempotency_key`, `rerun_count`, `interrupted` flag.
2. **`OnboardingCase`** (`operations_control/onboarding/case.py`) — 9 statuses
   (`draft → information_requested → awaiting_client → in_review →
   changes_required → ready_for_approval → approved → activated | withdrawn`) with
   its own `TRANSITIONS`, and — most relevant here — **`InformationRequest`**,
   with `open → sent → answered → accepted | rejected`, `responsible_party`,
   `requested_by/at`, `sent_at`, `due_date`, `response_note`, `responded_by/at`,
   `evidence: [{name, reference, received_at, received_by}]`, `reviewed_by/at`,
   `review_note`, and an `outstanding` property. Its docstring is almost a
   specification for the A2A case: *"what it needs is a governed record of what
   was asked, of whom, when, what came back, and who accepted it. A portal can
   post into the same record later."*
3. **`occ_agent/states.py`** — 19 states, transition assertion in the states
   module rather than in the caller, and the rule that *"the interpreter never
   decides a transition — it proposes an action, controls run, and the resulting
   state is asserted here."*

**What is missing:** all three are bound to onboarding or delivery of *one
client's data*. None is portfolio-scoped, none crosses organisations, and
`InformationRequest.items` are catalogue field references, not portfolio/loan
scopes.

### 9.2 Proposed minimal A2A workflow model

A new module, `trakt_core/dd.py` (contracts) + `operations_control/dd/` (store and
service), reusing `OpsStore` for persistence and hash-chained audit. **Five
objects, one state machine. Not a BPM engine — no process definitions, no
routing rules, no dynamic forms.**

#### `DD_REQUEST`

```python
@dataclass
class DDRequest:
    request_id: str                    # "ddr_<12hex>"
    engagement_id: str                 # groups a DD exercise
    tenant_id: str                     # whose data
    requester_organisation_id: str     # e.g. "buyer_a"
    responder_organisation_id: str     # e.g. "seller_a"
    resource_ref: str                  # "ERE/portfolio/acquired_001"
    category: str                      # DD_CATEGORIES
    subject: Dict[str, Any]            # {loan_ids[], canonical_fields[], as_of}
    question: str                      # one plain sentence
    rationale: str                     # why it matters (agent-authored)
    severity: str                      # blocking | material | informational
    due_date: str
    status: str                        # DD_REQUEST_STATUSES
    created_by_actor: str              # agent or human actor_id
    created_by_actor_type: str         # service | user
    created_at: str
    correlation_id: str                # ties to the tool calls that raised it
    schema_version: str
```

`DD_CATEGORIES = (missing_data, data_discrepancy, valuation_evidence,
document_request, covenant_clarification, performance_history, legal_title,
other)`

`DD_REQUEST_STATUSES = (draft, submitted, delivered, answered, accepted,
rejected, escalated, withdrawn, expired)`

```
  draft ──► submitted ──► delivered ──► answered ──┬─► accepted   (terminal)
    │           │             │                    ├─► rejected ──► delivered
    │           │             │                    └─► escalated ──► accepted
    │           │             └──────────────────────► expired    (terminal)
    └───────────┴──────────────────────────────────────► withdrawn (terminal)
```

Only two transitions may be made by an agent: `draft → submitted` (requester) and
`delivered → answered` (responder). `submitted → delivered` is Trakt's own
validation + permission step. `accepted`/`rejected` are the requester's
assessment. `escalated` always creates a `HumanEscalation`.

#### `DD_RESPONSE`

```python
@dataclass
class DDResponse:
    response_id: str
    request_id: str
    responder_organisation_id: str
    outcome: str                       # supplied | partially_supplied
                                       # | unavailable | refused | disputed
    narrative: str                     # plain language, no numbers not in values
    values: List[Dict[str, Any]]       # [{loan_id, canonical_field, value,
                                       #   provenance_envelope}]
    evidence: List[Dict[str, str]]     # [{kind, artefact_ref, content_hash}]
    unavailable_items: List[Dict[str, str]]   # [{loan_id, field, reason}]
    responded_by_actor: str
    responded_by_actor_type: str
    responded_at: str
    correlation_id: str
```

The critical rule, enforceable in the service: **every entry in `values` must
carry the provenance envelope from §8, produced by Trakt from governed data — not
authored by the responding agent.** The Seller Agent chooses *what to disclose*;
it never chooses *what the number is*. That single constraint is what makes this
demonstration institutionally meaningful rather than two chatbots agreeing.

#### `EXCEPTION`

Do **not** invent a new object. Reuse the existing shape:
`validate_business_rules` findings and `exception_db.findings` already have
`rule_id`, `severity`, `field_name`, `row_index`, `message`, `classification`,
`materiality`, `status`. Add one optional field, `dd_request_id`, linking an
exception to the DD request raised about it. An `Exception` in the A2A sense is
"a finding that has been escalated into the workflow", not a new type.

#### `DECISION`

```python
@dataclass
class DDDecision:
    decision_id: str
    engagement_id: str
    subject: Dict[str, Any]            # {level: loan|cohort|portfolio, ids[]}
    classification: str                # ACCEPT | PRICE_ADJUST | EXCLUDE
                                       # | HUMAN_REVIEW
    rationale: str
    basis: List[str]                   # tool call ids + dd request ids relied on
    price_adjustment: Optional[Dict]   # {basis_points | amount, currency}
    decided_by_actor: str
    decided_by_actor_type: str
    decided_at: str
    superseded_by: Optional[str]
    status: str                        # open | confirmed | superseded
```

`basis` is the point: a decision names the tool calls and DD requests it rests
on, so the audit trail is a graph rather than a narrative. `HUMAN_REVIEW`
automatically creates a `HumanEscalation`.

#### `HUMAN_ESCALATION`

```python
@dataclass
class HumanEscalation:
    escalation_id: str
    engagement_id: str
    trigger: str                       # dd_request_rejected | agent_uncertain
                                       # | policy_threshold | decision_human_review
                                       # | unavailable_blocking
    source_ref: str                    # request_id / decision_id / exception id
    summary: str
    assigned_to: str                   # role or principal
    status: str                        # open | acknowledged | resolved | dismissed
    resolution_note: str
    resolved_by: str                   # ALWAYS a named human
    resolved_at: str
```

#### `DD_ENGAGEMENT` (the container)

```python
@dataclass
class DDEngagement:
    engagement_id: str
    tenant_id: str
    buyer_organisation_id: str
    seller_organisation_id: str
    resource_ref: str
    investment_criteria: Dict[str, Any]   # thresholds the buyer was given
    status: str    # open | analysis | awaiting_responses | concluding
                   # | concluded | abandoned
    conclusion: Optional[Dict[str, Any]]
    opened_at: str
    concluded_at: str
```

**Persistence:** `OpsStore` under a new `dd/` prefix, with
`append_audit(client_id, event=…, actor=…, detail=…)` called on every transition
— giving the hash-chained, tamper-evident trail for free.

**What this deliberately is not:** no process definition language, no dynamic
forms, no SLA engine, no assignment rules, no parallel branches. Five documents
and one state machine.

---

## 10. Synthetic A2A architecture

### 10.1 Components

```
┌──────────────────────────────┐            ┌──────────────────────────────┐
│  BUYER AGENT                 │            │  SELLER AGENT                │
│  demo/agents/buyer.py        │            │  demo/agents/seller.py       │
│                              │            │                              │
│  system role: acquisition    │            │  system role: vendor         │
│    analyst for Buyer Co      │            │    data-room custodian       │
│  objective: classify every   │            │  objective: answer permitted │
│    loan; produce a portfolio │            │    requests accurately;      │
│    conclusion                │            │    disclose nothing outside  │
│  model: ANY (config)         │            │    the request scope         │
│                              │            │  model: ANY (config)         │
│  identity: org=buyer_a       │            │  identity: org=seller_a      │
│  scopes: portfolio:read,     │            │  scopes: portfolio:read,     │
│    mi:query, risk:read,      │            │    mi:query, risk:read,      │
│    dd:request                │            │    loan:read, document:read, │
│                              │            │    artefact:read, dd:respond │
│  tools: 1,2,3,5,6,7,8,9,10,  │            │  tools: 1,2,3,5,6,13,14,15,  │
│    11,12,13,14, raise_dd,    │            │    16,17, list_dd, respond_dd│
│    list_dd, record_decision  │            │                              │
└──────────────┬───────────────┘            └──────────────┬───────────────┘
               │  HTTPS + bearer                           │  HTTPS + bearer
               │  (client credentials, org=buyer_a)        │  (org=seller_a)
               ▼                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  TRAKT                                                                    │
│                                                                           │
│  /v1/agent/tools/{name}          agent_api.py  ── thin adapter            │
│         │                                                                 │
│         ▼                                                                 │
│  trakt_tools.execute_governed_tool(name, args, context)                   │
│    1 scope check              ExecutionContext.require_scope              │
│    2 resource authorisation   entitlement.authorise_resource_access       │
│    3 source approval          policy.evaluate_source_approval             │
│    4 deterministic execution  existing engines — UNCHANGED                │
│    5 envelope                 GovernedResult + provenance                 │
│    6 audit                    emit_audit_event + OpsStore.append_audit    │
│         │                                                                 │
│         ├──► deterministic estate  mi_query_executor · concentration_tests│
│         │                          period_change · analytics_lib          │
│         │                                                                 │
│         └──► DD service            operations_control/dd/                 │
│                 DDEngagement · DDRequest · DDResponse                     │
│                 DDDecision · HumanEscalation                              │
│                 state machine + permission asymmetry                      │
│                                                                           │
│  canonical: platform_canonical_typed.csv (from simulation/ generator)     │
│  storage:   OpsStore (blob or local) — hash-chained audit                 │
└──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  ┌───────────────────────────┐
                  │  HUMAN (operator / IC)     │
                  │  OCC React screen:         │
                  │    escalations, decisions, │
                  │    full audit trail        │
                  └───────────────────────────┘
```

### 10.2 The interaction protocol

Agents **never** exchange free text. Every crossing is a stored document:

```
Buyer Agent                     TRAKT                        Seller Agent
    │                             │                                │
    │ POST tools/evaluate_covenants                                │
    ├────────────────────────────►│  scope? entitled? approved?    │
    │◄────────────────────────────┤  GovernedResult + provenance   │
    │                             │                                │
    │ POST tools/data_completeness│                                │
    ├────────────────────────────►│                                │
    │◄────────────────────────────┤  27 loans missing valuation    │
    │                             │                                │
    │ POST tools/raise_dd_request │                                │
    │   {category: valuation_evidence,                             │
    │    subject: {loan_ids: [...27], fields: [current_valuation_amount]},
    │    severity: material}      │                                │
    ├────────────────────────────►│  validate scope ⊆ buyer grant  │
    │                             │  persist DDRequest (submitted) │
    │◄──── {request_id} ──────────┤  transition → delivered        │
    │                             │                                │
    │                             │  GET tools/list_dd_items       │
    │                             │◄───────────────────────────────┤
    │                             │  ONLY requests addressed to    │
    │                             │  seller_a, scope-filtered      │
    │                             ├───────────────────────────────►│
    │                             │                                │
    │                             │  POST tools/get_loan (×27)     │
    │                             │◄───────────────────────────────┤
    │                             ├───────────────────────────────►│
    │                             │                                │
    │                             │  POST tools/respond_dd_request │
    │                             │    {outcome: partially_supplied,
    │                             │     values: [22 loans],        │
    │                             │     unavailable_items: [5]}    │
    │                             │◄───────────────────────────────┤
    │                             │  Trakt REPLACES agent-supplied │
    │                             │  values with governed values + │
    │                             │  provenance; refuses any loan  │
    │                             │  outside the request scope     │
    │                             │  transition → answered         │
    │◄──── notification ──────────┤                                │
    │                             │                                │
    │ POST tools/list_dd_items    │                                │
    ├────────────────────────────►│                                │
    │◄─ 22 supplied, 5 unavailable┤                                │
    │                             │                                │
    │ POST tools/evaluate_covenants (rerun with new data)          │
    ├────────────────────────────►│                                │
    │◄────────────────────────────┤                                │
    │                             │                                │
    │ POST tools/record_decision × N                               │
    │   {classification: EXCLUDE, subject: {loan_ids: [5]},        │
    │    basis: [ddr_…, tool_call_…]}                              │
    ├────────────────────────────►│  persist DDDecision            │
    │                             │  HUMAN_REVIEW → HumanEscalation│
    │◄────────────────────────────┤                                │
```

### 10.3 Where state lives

| State | Store | Why |
|---|---|---|
| Canonical portfolio | `platform_canonical_typed.csv` + `snapshot/store.py` | Existing; content-hashed, idempotent |
| Entitlements | `config/entitlements.yaml`, `config/organisations.yaml`, `config/resources.yaml` | Existing model; server-side config only, never request data |
| DD documents | `OpsStore` under `dd/{engagement_id}/` | Reuses hash-chained audit and blob/local abstraction |
| Tool-call audit | `trakt.audit` log **plus** `OpsStore.append_audit` per call | The log is for operations; the hash chain is the evidence |
| Agent conversation | Agent-side only, never in Trakt | Trakt stores facts and decisions, not reasoning traces |

### 10.4 How calls are audited

Every `execute_governed_tool` emits both:

- a `trakt.audit` line via the existing `emit_audit_event(result)` — capability,
  request id, correlation id, tenant, organisation, actor, actor type, channel,
  portfolio, snapshot id, outcome, duration, error code, with the `_FORBIDDEN_KEYS`
  denylist ensuring no answer body or borrower data leaks into logs;
- an `OpsStore.append_audit` record under the engagement, giving a
  hash-chained, `verify_audit_chain`-checkable sequence.

`correlation_id` threads a whole DD exchange: the Buyer Agent's analysis calls,
the DD request, the Seller Agent's retrieval calls, the response, and the
resulting decision all share it. That is what makes "produce the audit trail for
this decision" a query rather than a reconstruction.

### 10.5 How agents remain model-agnostic

- The agents are **harnesses, not models**: a system prompt, a tool list, and a
  loop. `demo/agents/runner.py` takes `--provider anthropic|openai|azure` and a
  model id from config.
- Tool schemas are JSON Schema from the registry, translated per provider at the
  harness boundary. Nothing provider-specific enters Trakt.
- Trakt's own internal LLM use (Tier 7 mapper, MI interpreter) is unrelated and
  unchanged; the demo does not depend on it.
- A regression test asserts the demo completes with a **stubbed deterministic
  agent** that calls a fixed tool sequence — proving the workflow is a property
  of Trakt, not of any model.

---

## 11. Gap register

Ranked by severity. Severity = impact on the stated strategy, not defect count.

| # | Gap | Severity | Existing component to extend | Work required | When | Dependencies |
|---|---|---|---|---|---|---|
| G1 | No typed, agent-callable tool registry; only one NL capability | **Critical** | `mi_agent_api/mi_service.py` pattern; `trakt_core/envelope.py` | New `trakt_tools/registry.py` + 12 thin handlers over existing engines | **NOW** | — |
| G2 | No machine identity; entitlement config dormant (all `*.example.yaml`) | **Critical** | `trakt_core/entitlement.py`, `organisation.py`, `copilot_auth.py` | Materialise 5 config files; add `agent_auth.py` + `context_from_service_principal` | **NOW** | — |
| G3 | No external agent interface (no REST tool API, no MCP) | **Critical** | `deploy/copilot-agent/trakt-copilot-openapi.yaml` pattern | `agent_api.py` (~300 lines) + generated OpenAPI | **NOW** | G1, G2 |
| G4 | No loan-level read path (`get_loan`) | **High** | Entity model (E1) + `datasets.py` | New capability + `config/system/entity_model.yaml` | **NOW** | E1 |
| G5 | No runtime value-level provenance (`explain_value`) | **High** | `lineage_tracker.py` artefacts, `provenance.py`, validation results | New capability joining existing artefacts at request time | **NOW** | G4 |
| G6 | ~25 UI routes bypass the governed capability layer (no `GovernedResult`, no scope gate; risk/forecast routes explicitly ungated) | **High** | `mi_service.py` pattern | Convert the 6 tools' worth (`risk_limits`, `concentration_tests`, `period_change`, `cohorts`, `evolution`, `forecast`) into governed capabilities; leave the React feeds alone | **NOW** for the 6; NEXT for the rest | G1 |
| G7 | No DD workflow objects; no portfolio-scoped request/response | **High** | `onboarding/case.InformationRequest`, `contracts.WorkflowRun` | New `trakt_core/dd.py` + `operations_control/dd/` (§9) | **NEXT** | G1, G2 |
| G8 | Duplicated analytics: `analytics/` vs `mi_agent*`; `mi_prep` vs `funded_prep`; three `pipeline_prep`; percent scale decided in 4 places | **High** | `docs/spine_audit_single_source_of_truth.md` already scopes it | Ring-fence `analytics/` as legacy (import guard test); consolidate percent-scale on the server | **NEXT** | — |
| G9 | No entity/relationship declaration in the registries | **Medium-High** | `config/system/fields_registry.yaml` | Add `entity`/`entity_role`/`references` keys + `entity_model.yaml` (~8 entities) | **NOW** (small) | — |
| G10 | No outbound seam — Trakt cannot notify an agent | **Medium-High** | `trakt_notifications/outbox.py` (durable, at-least-once, idempotent) | Generalise the outbox to a capability-level webhook/callback | **NEXT** | G7 |
| G11 | `loan:read`, `document:read`, `dd:request`, `dd:respond` capabilities do not exist | **Medium** | `trakt_core/context.KNOWN_CAPABILITIES` | Four constants + grant-config support | **NEXT** | G2 |
| G12 | Agent tool calls are not queryably audited (log line only) | **Medium** | `OpsStore.append_audit` (hash-chained) | Write a second audit record per tool call under the engagement | **NEXT** | G1, G7 |
| G13 | No `on_behalf_of` — cannot distinguish "agent acting for Jane" from "agent" | **Medium** | `ExecutionContext`, `AuditMetadata` | Two optional fields + audit projection | **NEXT** | G2 |
| G14 | No idempotency on the capability/tool path | **Medium** | `snapshot/store.py`, `annex_delivery_agent`, `intake.py` all do it | Accept `Idempotency-Key`; required for write tools (`raise_dd_request`, `record_decision`) | **NEXT** (writes) | G7 |
| G15 | Calculation methodology/version not in the envelope | **Medium** | `period_change.CALCULATION_VERSION`, concentration `library_version` | Two optional fields on `ProvenanceRef` | **NEXT** | — |
| G16 | Insight generation duplicated client-side (`frontend/.../lib/insights.ts`, 380 lines: shares, stdev, top-3 share, movement) vs `mi_agent_api/insight_engine.py` | **Medium** | `insight_engine.py` | Make the server the source; React renders | **NEXT** | G8 |
| G17 | Only one artefact type flows through the governed path; the other four resolve directly | **Medium** | `artefacts.get_investor_pack`, `copilot_artifacts` registry | Generalise `artefacts.py` to any registered type | **NEXT** | G2 |
| G18 | `exception_db` (SQLite) is disconnected from the governed path and the OCC store | **Medium** | `exception_db.py`, `aggregate_validation_results.py` | Read-through for `list_validation_exceptions`; decide one home | **NEXT** | G6 |
| G19 | Only one LLM provider adapter exists (Anthropic); `llm_policy.provider` has one branch | **Low-Medium** | `AnthropicMIInterpreterClient(Protocol)`, `llm_policy.py` | Second adapter, only if a client requires it — the Protocol is already right | **LATER** | — |
| G20 | Field-level and document-level authorisation do not exist | **Low** | `Grant`, `ResolvedResource.predicate()` | Field allow-list on grants | **LATER** | G11 |
| G21 | Indexed LTV / HPI is `interface_only` — declared, never simulated | **Low** | `concentration_test_library.yaml`, `ExternalIndexProvider` | Configure an approved index feed | **LATER** | — |
| G22 | No `eligibility` evaluator behind the declared concept | **Low** | `business_semantics_registry` taxonomy | Genuine new business logic, per-client | **LATER** | — |
| G23 | Single-tenant namespace default: without `config/tenancy.yaml`, the tenant owns any well-formed selector | **Low** *(if G2 done)* | `trakt_core/tenancy.py` | Closed by G2 | **NOW** (via G2) | G2 |

---

## 12. Recommended implementation sequence

Sequenced so each phase is independently demonstrable and none is wasted if the
next is deferred.

### Phase 1 — External Agent Proof

**Goal:** an independent Claude or OpenAI agent authenticates to Trakt and uses
12–15 deterministic credit tools against a portfolio, with every call
authorised, provenanced and audited.

| # | Task | Touches |
|---|---|---|
| 1.1 | Create `trakt_tools/registry.py`: `ToolSpec(name, version, description, input_model, output_model, required_scope, handler)` + `execute_governed_tool(name, args, context)` running scope → resource authorisation → source approval → handler → `GovernedResult` → audit | NEW |
| 1.2 | Implement 12 thin handlers wrapping existing engines: tools 1,2,3,5,6,7,8,9,10,11,12,13 from §5. **No calculation written.** | NEW handlers; existing engines untouched |
| 1.3 | Add `entity`/`entity_role`/`references` to the field registry for ~40 key fields; add `config/system/entity_model.yaml` (~8 entities) | `config/system/` |
| 1.4 | Implement `get_loan` (tool 15) assembling a nested loan object from the canonical row via the entity model, with per-field compact provenance stubs | NEW capability |
| 1.5 | Implement `explain_value` (tool 16) joining `field_lineage.json`, the canonical row, validation results and the snapshot manifest into the §8 envelope | NEW capability |
| 1.6 | Extend `ProvenanceRef` with optional `calculation: {method, version}` and `evidence: [{kind, ref}]` (additive — no consumer breaks) | `trakt_core/envelope.py` |
| 1.7 | Add `SCOPE_LOAN_READ` / `loan:read` to `KNOWN_CAPABILITIES`; gate tools 15-16 on it; **do not** add it to `DEFAULT_MI_SCOPES` | `trakt_core/context.py` |
| 1.8 | Write `mi_agent_api/agent_auth.py` (fork the `copilot_auth` validation) and `identity.context_from_service_principal(...)` producing `actor_type=ACTOR_SERVICE, channel=CHANNEL_ENTERPRISE_AGENT` with **narrowed** scopes from the organisation record | NEW + `identity.py` |
| 1.9 | Write `mi_agent_api/agent_api.py`: `GET /v1/agent/tools` (JSON Schemas) + `POST /v1/agent/tools/{name}`. Thin — build context, invoke, return `result.to_dict()` | NEW |
| 1.10 | Materialise real `config/tenancy.yaml`, `organisations.yaml`, `resources.yaml`, `entitlements.yaml`, `principals.yaml` for the existing tenant plus one test agent organisation | `config/` |
| 1.11 | Generate `deploy/agent-api/trakt-agent-openapi.yaml` from the registry; add a lock-step test (mirroring `test_copilot_package.py`) | NEW + tests |
| 1.12 | Convert the 6 tool-backing routes into governed capabilities so the tool and the React route share one implementation (`risk_limits`, `concentration_tests`, `period_change`, `cohorts`, `evolution`, `forecast`) | `mi_agent_api/` |
| 1.13 | Tests: per-tool contract tests; an unentitled organisation is refused on every tool; a scope-missing caller is refused; audit is emitted on refusal as well as success; the dependency-direction test still passes | `tests/` |
| 1.14 | A reference client: `demo/agents/reference_client.py` — a real Claude/OpenAI harness calling the tool list, plus a deterministic stub client used in CI | NEW |

**Exit criterion:** an external agent, holding only client-credentials for
`organisation_id=test_agent`, completes a portfolio review across ≥10 tools; every
call appears in the audit log with organisation, resource and outcome; a call
against an ungranted portfolio is refused indistinguishably from a nonexistent
one.

### Phase 2 — Synthetic Buyer Agent

**Goal:** a Buyer Agent performs an autonomous portfolio review / acquisition-DD
analysis against a synthetic portfolio, unassisted.

| # | Task | Touches |
|---|---|---|
| 2.1 | Generate the demo portfolio with `simulation/` — deterministic seed, with *deliberately planted* defects: missing valuations on a cohort, an LTV outlier band, a concentration breach, a stale reporting date on one source portfolio | `simulation/` config only |
| 2.2 | Implement `data_completeness` (tool 14) reading `config/mna/diligence_scorecard.yaml`, which is currently declared and unread | NEW handler |
| 2.3 | Implement `list_validation_exceptions` (tool 13) as a governed read over run artefacts and `exception_db` | NEW handler |
| 2.4 | Implement `DDDecision` + `record_decision` tool with `ACCEPT / PRICE_ADJUST / EXCLUDE / HUMAN_REVIEW` and a `basis` list of tool-call and DD-request ids | `operations_control/dd/` |
| 2.5 | Build the Buyer Agent harness: system role, investment criteria from config, provider-agnostic tool loop, bounded iterations | `demo/agents/buyer.py` |
| 2.6 | Second audit sink: `OpsStore.append_audit` per tool call under an engagement id, so the trail is hash-chained and queryable | `operations_control/` |
| 2.7 | An OCC React screen listing engagements, decisions and the audit trail | `frontend/operations-control-ui/` |
| 2.8 | Regression: the same run with a deterministic stub agent produces the same decisions — proving the workflow is Trakt's, not the model's | `tests/` |

**Exit criterion:** given criteria and a portfolio, the Buyer Agent produces a
per-loan classification and a portfolio conclusion in which every number is
traceable to a tool call, and no number was authored by the model.

### Phase 3 — Buyer ↔ Seller Synthetic A2A

**Goal:** two independently permissioned agents exchange structured DD requests
and responses through Trakt.

| # | Task | Touches |
|---|---|---|
| 3.1 | Implement `trakt_core/dd.py` — the five contracts and the state machine from §9 | NEW |
| 3.2 | Implement `operations_control/dd/` — store on `OpsStore`, service enforcing transitions, `IllegalTransition` on violation | NEW |
| 3.3 | Add `dd:request` / `dd:respond` capabilities; grant asymmetrically to `buyer_a` / `seller_a` in `config/entitlements.yaml` | `trakt_core/`, `config/` |
| 3.4 | Tools `raise_dd_request`, `list_dd_items`, `respond_dd_request`, with `Idempotency-Key` support on the two writes | NEW handlers |
| 3.5 | **The disclosure control:** `respond_dd_request` re-resolves every value from governed data and attaches the §8 provenance envelope. A value the responding agent supplies that disagrees with governed data is rejected, not stored. Scope outside the request is refused. | `operations_control/dd/` |
| 3.6 | Build the Seller Agent harness with a *different* system role, objective, tool list and permission set | `demo/agents/seller.py` |
| 3.7 | Generalise `trakt_notifications/outbox.py` into a capability-level callback so Trakt can notify an agent a request was answered (closing the named outbound gap) | `trakt_notifications/` |
| 3.8 | Implement `HumanEscalation` + the OCC escalation queue screen | `operations_control/`, frontend |
| 3.9 | The 10-step scripted demo, runnable end to end with `--provider` selecting the model, plus a stub-agent CI variant | `demo/` |
| 3.10 | Security tests: the Seller Agent cannot read the Buyer's decisions; the Buyer cannot read seller loans outside a request's answered scope; neither can transition a state the other owns | `tests/` |

**Exit criterion:** the full 10-step workflow completes; the audit trail
reconstructs, for any decision, the tool calls and DD exchanges it rested on;
removing the Buyer's `dd:request` grant fails the run at exactly the right step.

### Phase 4 — Real Portfolio Pilot

**Goal:** run the synthetic agent workflow against real customer data, without
requiring the customer to operate its own agent.

| # | Task |
|---|---|
| 4.1 | Run only the Buyer-side agent against a real client portfolio; the "Seller Agent" is replaced by the OCC human queue answering DD requests through the existing UI — the same `DDRequest` objects, a human responder |
| 4.2 | Verify source-approval policy on the real path: `platform_canonical` base, `production` runtime mode, no fixture exemption |
| 4.3 | Verify the Azure deployment requirements D1–D4 from `docs/governed_capability_architecture.md` for the new agent API host |
| 4.4 | Rate limiting and per-agent budget caps (mirroring `max_api_calls_per_session` in `config/system/config_agent.yaml`) |
| 4.5 | Consolidate G8/G16 before agent-visible numbers can diverge from UI numbers in front of a client |
| 4.6 | Operator runbook: how to grant, revoke and audit an agent; how to answer a DD request as a human |
| 4.7 | Retention and export: an engagement audit pack (extend `export_audit_pack.py`) |

**Exit criterion:** a real portfolio review produces a decision pack a credit
committee would accept, with every figure provenanced, and no agent ever reached
data outside its grants.

### Phase 5 — External Client Agent

**Goal:** a client's own Copilot / Claude / OpenAI / proprietary agent connects to
Trakt.

**Prerequisites only — deliberately not over-specified:**

- Phases 1–4 complete and stable.
- Organisation onboarding through `operations_control/access_admin/` (the
  `AccessChangeSet → DRAFT → named-human confirm → ACTIVE` lifecycle already
  exists; the new work is a screen for agent registration).
- Credential lifecycle: issuance, rotation, revocation, expiry — an operational
  capability Trakt does not have today.
- The MCP server (`trakt_mcp/`) generated from the same registry, for clients
  whose agent frameworks prefer it.
- A published, versioned tool contract with a deprecation policy — the tool
  registry must carry `version` per tool from Phase 1 precisely so this is
  possible later.
- Per-organisation rate limits, quotas and usage reporting.
- Contractual and DPA framing for a counterparty agent reading client data.

---

## 13. What NOT to build

Each item below is named because something in this review positively supports
excluding it — not as generic caution.

| Do not build | Why, from this review |
|---|---|
| **A proprietary foundation model** | Nothing in the architecture needs one. The LLM's only jobs are interpretation (`mi_agent/interpreter/`, already behind a `Protocol` and constrained to emit validated spec JSON) and tool selection. The deterministic path — `mi_query_executor`, `concentration_tests`, `period_change` — runs with no model at all. A proprietary model would add cost and eliminate the model-agnosticism that is the stated design goal. |
| **A graph database / triple-store migration** | §4.5. The access pattern is set-based aggregation over a wide typed frame; relationship depth is 2–3 hops; every existing control (validation rules, `ResolvedResource.predicate()`, attribution checks) is written against a columnar frame and would be invalidated. Declaring entities in the existing YAML gets the entire benefit. |
| **A custom inter-agent communications protocol** | The agents never talk to each other. Every exchange is a stored, permissioned document through Trakt (§10.2). Inventing a wire protocol would add a surface with no security model, when HTTPS + bearer + a document store already covers it. |
| **A generic BPM / workflow engine** | Three enforced state machines already exist and each is ~50 lines of constants plus a transition dict. §9 needs one more of the same shape. A BPM engine would introduce process definitions, dynamic routing and a second source of truth for state — against a requirement whose whole point is that state transitions are asserted by controls, not chosen by an interpreter. |
| **A multi-agent orchestration framework** (LangGraph, CrewAI, AutoGen and similar) | The demo needs two harnesses with a tool loop, ~200 lines each. A framework would couple Trakt's demo to a third-party abstraction, make provider-agnosticism harder rather than easier, and obscure the property being demonstrated — that the *workflow* is Trakt's, not the framework's. |
| **Sophisticated ABAC / a policy engine (OPA, Cedar)** | §7.3. The two-axis model (scopes ∩ per-resource grants) plus `ResolvedResource.predicate()` for row-level filtering already covers everything through Phase 4, and it is tested. A policy engine would duplicate `authorise_resource_access` and split the security model across two languages. |
| **Bespoke Buyer / Seller LLMs, or fine-tuning** | The agents' differentiation is *role, objective, permission set, tool list and information access* — all enforced server-side by Trakt. If two agents on the same base model behave differently only because Trakt gives them different tools and data, that is the demonstration. Fine-tuning would weaken the claim, not strengthen it. |
| **Autonomous bid or trade execution** | Explicitly out of scope in the brief, and correct: `DDDecision` with `HUMAN_REVIEW` and `HumanEscalation` requiring a named human resolver is the right terminal state. The `access_admin` precedent is instructive — the OCC Agent may only ever *propose* a grant; only a named human activates it. Apply the same rule to anything with money attached. |
| **A second copy of any calculation for agent use** | G8 already documents what happens when a responsibility is implemented twice — header detection, numeric parsing, percent scaling and chart-key derivation have all drifted. Every tool in §5 wraps an existing implementation; where two exist (`analytics/risk_monitor.py` vs `mi_agent/risk_monitor/`), pick one and ring-fence the other. |
| **A full REST rewrite of the ~25 UI routes** | Only six back the proposed tools and only those should become governed capabilities in Phase 1. The rest are React panel feeds; converting them would be effort spent on a surface no agent needs. |
| **A per-cell provenance store** | §8.3. Materialising a provenance envelope for every value would multiply storage and add a second data model. The inputs are already persisted; `explain_value` joins them on demand, and the cost is paid only when an agent asks "why". |
| **An `eligibility` engine before the demo** | `eligibility` is a declared analytical concept with no evaluator behind it. Building one is genuine new per-client business logic, not an exposure task, and would be a plausible-looking way to spend the demo's budget on something the demo does not need. |

---

## 14. Final recommendation

### Is the current architecture suitable for this strategy?

**Yes, and unusually so.** Three properties that are hard to retrofit are already
present and enforced by tests:

1. **Reasoning and execution are already separated.** `mi_agent/interpreter/`
   lets the LLM propose an `MIQuerySpec` and nothing else; the spec is normalised
   and validated before it can run, and an ambiguous interpretation forces a
   clarification. `occ_agent/service.py` gives the interpreter a bounded typed
   tool surface with no general-purpose escape hatch. The Tier 7 field mapper
   requires human confirmation and learns a deterministic alias so the LLM is not
   consulted twice for the same question. The principle the strategy asks for is
   not aspirational here — it is implemented, three times, in different places.

2. **Governance is interface-neutral by construction.** `trakt_core` cannot
   import FastAPI, and a subprocess test proves that *calling* a capability does
   not load one. That is why an MCP server, a REST tool API and the existing React
   route can share one authorisation path rather than three.

3. **The permission model already asks the A2A question.** `organisation ×
   resource × capability`, with explicit grants rather than inferred roles, and a
   refusal for resources the data cannot partition. Buyer Agent and Seller Agent
   are two rows of configuration in a model that exists.

### Evolution or rewrite?

**Evolution, decisively.** No component reviewed needs replacing. The work is:
expose (G1, G3), activate (G2), extend (G4, G5, G7, G9) and consolidate (G8, G16).

### The smallest next engineering experiment

**One tool, end to end, in a week.**

Take `evaluate_covenants`. Write `trakt_tools/registry.py` with a single entry
wrapping `concentration_tests.evaluation.evaluate_active_tests`. Write
`agent_auth.py` and one route, `POST /v1/agent/tools/evaluate_covenants`.
Materialise `config/organisations.yaml` and `config/entitlements.yaml` with one
test agent organisation granted `risk:read` on one portfolio. Point a Claude
agent at it with a JSON Schema.

That single exercise tests every load-bearing assumption in this review: that an
external agent can authenticate as a service, that entitlements resolve and
refuse correctly when switched on, that `GovernedResult` is a usable agent
response, that a deterministic engine wraps cleanly without modification, and
that the audit line is sufficient. If it works, Phases 1–3 are execution. If it
does not, it fails in a week rather than a quarter.

### The highest-value architectural change

**Value-level provenance as a service (`explain_value`, G5) — with the tool
registry (G1) as its vehicle.**

This is the change that converts Trakt from a system that produces numbers into
one that produces *facts*. Everything else in this review is exposure of existing
capability; this is the one that changes what Trakt *is*. It is also the answer
to why an institution would route a counterparty's agent through Trakt at all
rather than exchanging spreadsheets: a number with a verifiable provenance
envelope is a materially different asset from a number.

And the cost is low, because the inputs already exist. `lineage_tracker.py`
already emits field and value lineage; `provenance.py` already stamps every row;
`delta_manifest.py` already hashes; validation already records rule outcomes.
`explain_value` is a join, not a data model.

### The biggest technical risk

**Divergence between the agent path and the human path (G8, G16).**

Two live analytics implementations exist today — `analytics/mi_prep.py` vs
`mi_agent_api/funded_prep.py`, with different numeric parsers, bucketing and
LTV/age derivation — plus percent scale decided in Python and re-decided
differently in React, and 380 lines of client-side insight generation duplicating
`insight_engine.py`. A human noticing a 0.2pp discrepancy between a dashboard and
a deck raises a ticket. An agent that raises a DD request with a counterparty on
the basis of the wrong one creates an institutional problem, and the audit trail
will faithfully record that Trakt asserted it.

The mitigation is cheap and should be done in Phase 1, not Phase 4: make each
Phase-1 tool and its corresponding React route share one governed implementation
(task 1.12), add an import guard test ring-fencing `analytics/` as legacy, and
move percent-scale resolution to the server. The consolidation is already scoped
in `docs/spine_audit_single_source_of_truth.md` — it needs prioritising, not
analysing.

### The biggest unnecessary-distraction risk

**Building an ontology layer and a graph database because the word "ontology"
appears in the requirement.**

Trakt already has a two-layer semantic model — a 499-field canonical registry
with regulatory code mappings, and a 242-field business-semantics registry with a
controlled taxonomy of concepts, roles, temporality, directionality, default
aggregations and weight fields. What is missing is roughly **eight entity
declarations and a handful of foreign-key annotations in YAML** — perhaps a day
of work plus review.

The failure mode is spending a quarter on a knowledge graph, a mapping layer and
a query language, ending with the same analytical capability, a slower system,
and a second source of truth for what a field means. The second-biggest
distraction is the same instinct applied to workflow: reaching for a BPM engine
when the codebase already contains three correct, small, enforced state machines
and needs a fourth of the same shape.

The discipline that has served this codebase well — extend the registry, wrap the
engine, reuse the store, add a state machine that fits on one screen — is the
discipline that will get it to A2A.

---

## Appendix — Principal source references

| Area | Files |
|---|---|
| Governance core | `trakt_core/{context,tenancy,organisation,principal,resource,entitlement,policy,runtime,errors,envelope,audit,portfolio}.py` |
| Governed capabilities | `mi_agent_api/{mi_service,artefacts,dependencies,identity,auth,copilot_auth,copilot_actions,presenters}.py` |
| API surface | `mi_agent_api/app.py`; `operations_control/api/app.py`; `operations_control/occ_agent/api.py`; `deploy/copilot-agent/trakt-copilot-openapi.yaml` |
| Canonical model | `config/system/fields_registry.yaml`; `config/business_semantics_registry.yaml`; `mi_agent/mi_semantics_field_registry.yaml`; `config/system/canonical_derivations.yaml`; `config/system/aliases_*.yaml` |
| Pipeline | `engine/gate_1_alignment/{semantic_alignment,agent_orchestrator,llm_mapper_agent}.py`; `engine/gate_2_transform/{canonical_transform,lineage_tracker,delta_manifest}.py`; `engine/gate_3_validation/*.py`; `engine/gate_4_projection/*.py`; `engine/gate_5_delivery/*.py`; `engine/{provenance,platform_assembler,assembler_agent}.py` |
| Calculations | `mi_agent/mi_query_executor.py`; `mi_agent/concentration_tests/{library,metrics,evaluation,forward,matching}.py`; `mi_agent/period_change/*.py`; `mi_agent/risk_monitor/*.py`; `analytics_lib/*.py`; `mi_agent_api/{risk_limits,forecast_extrapolation,cohorts,evolution}.py` |
| Interpretation | `mi_agent/interpreter/{deterministic,anthropic,prompt,evaluator,runtime_bridge}.py`; `mi_agent/{mi_query_spec,mi_query_validator,mi_spec_validation}.py` |
| Workflow / OCC | `operations_control/{contracts,engine,stores,intake}.py`; `operations_control/onboarding/{case,service}.py`; `operations_control/occ_agent/{service,states,interpretation,execution}.py`; `operations_control/access_admin/*.py` |
| Data quality | `exception_db.py`; `exception_queue.py`; `ingest_violations.py`; `config/mna/diligence_scorecard.yaml` |
| Synthetic data | `simulation/`; `demo_platform/`; `synthetic_demo/` |
| Existing architecture docs | `docs/governed_capability_architecture.md`; `docs/spine_audit_single_source_of_truth.md`; `docs/mi_analytics_architecture_current_state_audit.md`; `docs/business_semantics_registry_review.md`; `docs/copilot_v1_implementation.md` |
| Tests | `tests/test_governance_*.py` (11 suites); `tests/test_governed_*.py`; 225 test modules total |
