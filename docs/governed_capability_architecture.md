# Trakt governed capability architecture

How a question or an artefact request becomes a governed answer, and what a new
interface has to write in order to join in.

Audience: engineers adding an interface, and reviewers checking the security
model. It documents the code as it is, not an intended end state.

---

## 1. The governed call flow

```
  React MI Agent        M365 Copilot         Python / job / CLI      (future adapters)
  POST /mi/query        POST /v1/copilot/    direct import           Teams · MCP · agent
       │                     mi/query             │                       │
       ▼                     ▼                    ▼                       ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ INTERFACE IDENTITY ADAPTER            mi_agent_api/identity.py                 │
 │   Easy Auth principal ─┐                                                       │
 │   Entra bearer token  ─┼──►  ExecutionContext(tenant, actor, channel,          │
 │   trusted in-process  ─┘                      scopes, request_id, correlation) │
 │   · tenant is DEPLOYMENT CONFIG, never a request field or a token claim        │
 │   · in production, refuses to trust the platform header when Azure platform    │
 │     authentication is not enabled (identity.require_trustworthy_platform_auth) │
 └───────────────────────────────────┬───────────────────────────────────────────┘
                                     ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ GOVERNED CAPABILITY                                                            │
 │   mi.question.answer          mi_agent_api/mi_service.py                       │
 │   artefact.investor_pack.get  mi_agent_api/artefacts.py                        │
 │                                                                                │
 │   1. scope check            context.require_scope(...)                         │
 │   2. portfolio authorisation trakt_core/tenancy.authorise_portfolio_access     │
 │   3. source approval        trakt_core/policy.evaluate_source_approval         │
 │   ── nothing above touches data; every check precedes the first read ──        │
 │   4. analytical execution   (unchanged: parser → validator → executor →        │
 │                              chart factory → adapters)                         │
 └───────────────────────────────────┬───────────────────────────────────────────┘
                                     ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ DOMAIN + INFRASTRUCTURE                                                        │
 │   mi_agent/mi_query_executor.py   deterministic calculation (untouched)        │
 │   analytics_lib/                  buckets · stratify · concentration           │
 │   mi_agent_api/datasets.py        dataset resolution — NO web framework        │
 │   apps/blob_trigger_app/storage   blob:// ↔ filesystem abstraction             │
 └───────────────────────────────────┬───────────────────────────────────────────┘
                                     ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ GovernedResult[T]                 trakt_core/envelope.py                       │
 │   capability · schema_version · status · request_id · correlation_id           │
 │   tenant_id · portfolio_id · snapshot · result · warnings                      │
 │   policy · provenance · audit · error                                          │
 └───────────────────────────────────┬───────────────────────────────────────────┘
                                     ▼
              mi_agent_api/presenters.py  →  React payload (unchanged + governance)
                                          →  Copilot action response
              trakt_core/audit.py         →  one structured audit line
```

### Package boundaries

| Package | Rule |
|---|---|
| `trakt_core/` | Contracts and policy. No FastAPI, no Azure, no pandas. Importable anywhere. |
| `mi_agent_api/datasets.py` | Dataset resolution. Imports storage and pandas; **no web framework**. |
| `mi_agent_api/mi_service.py`, `artefacts.py` | Governed capabilities. No FastAPI, no `app` import. |
| `mi_agent_api/app.py`, `copilot_actions.py` | Interface adapters. The only modules that import FastAPI. |
| `mi_agent/`, `analytics_lib/`, `engine/` | Domain. Unchanged by this work. |

`tests/test_governance_dependency_direction.py` enforces the direction, including
a subprocess check that importing and *calling* the capability never loads
FastAPI.

---

## 2. Adding a future interface

Every adapter is the same three steps. None of them may re-implement governance.

```python
# 1. turn your verified identity into a trusted context
context = ExecutionContext(
    tenant_id=...,          # from deployment config / your service registration
    actor_id=...,           # the authenticated user or service
    actor_type=ACTOR_SERVICE,
    channel=CHANNEL_ENTERPRISE_AGENT,
    scopes=DEFAULT_MI_SCOPES,
    correlation_id=inbound_trace_id,     # optional
)

# 2. translate your transport payload into the capability request
request = MiQueryRequest(question=..., portfolio_id=..., filters=...)

# 3. call the capability and present the result
result = execute_governed_mi_query(request, context, build_dependencies())
return my_transport_shape(result)
```

`tests/test_governance_artefacts_and_envelope.py::enterprise_agent_endpoint` is a
working example, exercised by two tests that prove the adapter inherits tenant
authorisation and the source-approval policy without writing either.

### Client enterprise-agent endpoint

1. Authenticate the service (Entra client credentials, mTLS, or a signed
   assertion) — reuse `copilot_auth` as the pattern.
2. Map the validated service identity to `ExecutionContext` with
   `actor_type=ACTOR_SERVICE`, `channel=CHANNEL_ENTERPRISE_AGENT`.
3. Add a route module that calls `execute_governed_mi_query` and returns
   `result.to_dict()` — the governed envelope is already machine-readable.
4. Add the tenant to `config/tenancy.yaml` with its portfolio allow-list.

No changes to `mi_service`, `datasets`, `trakt_core` or any calculation.

### Agent-to-agent event consumer

Same three steps, plus: build the context with the event's correlation id so the
audit trail joins up, and use an idempotency key on any artefact generation the
handler triggers. Note the gap in §6 — there is no outbound notification seam yet.

### MCP adapter

Same three steps per tool. Return `result.to_dict()`; it already excludes storage
paths and stack traces. Gate each tool on a scope so an MCP client cannot reach a
capability its token does not carry.

---

## 3. Security model

### Trusted source of tenant identity

| Channel | Identity mechanism | Tenant source |
|---|---|---|
| React | Azure Easy Auth / SWA injects `X-MS-CLIENT-PRINCIPAL`; `auth.py` decodes it | `MI_AGENT_CLIENT_ID`, else the client in `MI_AGENT_PLATFORM_URI` |
| Copilot | Entra bearer token validated against the issuing directory's JWKS (`copilot_auth.py`) | same deployment config |
| Internal | `ExecutionContext.for_internal(...)` — the process is inside the trust boundary | caller-declared |

The tenant is **never** taken from `portfolio_id`, `client_id` or a token claim.
`tests/test_governance_context_and_tenancy.py::test_tenant_is_never_taken_from_a_principal_claim`
locks that in.

### Organisation identity — who is asking

A context now carries a second identity, and the two must not be conflated:

| Field | Question it answers | Source |
|---|---|---|
| `tenant_id` | **Whose data is served?** | Deployment configuration. Unchanged. |
| `organisation_id` | **Who is asking?** | The caller's signature-verified Entra directory, mapped through `trakt_core/organisation.py` |
| `microsoft_tenant_id` | Which Entra directory did the token come from? | The validated `tid` claim |

They are separate because they stop being the same party as soon as a warehouse
funder, investor or servicer signs in to *its own* directory to ask about someone
else's book. A validated `tid` selects an **organisation** and never a tenant.

Two modes, chosen by whether `config/organisations.yaml` exists (see
`config/organisations.example.yaml`):

* **compatibility** (no file) — the current ERE shape. No organisation identity
  is claimed, `resolve_organisation` returns `None`, and every existing caller is
  unchanged.
* **organisation mode** (file present) — an unregistered or disabled directory is
  refused (`ORGANISATION_NOT_REGISTERED` / `ORGANISATION_DISABLED`, both 403) on
  any path carrying a validated directory. There is no permissive fallback, and a
  config file that cannot be trusted leaves the registry strict-but-empty rather
  than reopening the deployment.

`copilot_auth` validates against a **set** of accepted directories — the
comma-separated `TRAKT_COPILOT_ENTRA_TENANT_ID` plus every enabled organisation's
directories. The `tid` claim is read unverified only to *select* which
allow-listed directory's JWKS to verify the signature against; nothing in the
token is trusted until that verification succeeds.

Organisations are **identity only**. They carry no entitlements, resources or
capabilities yet, so registering one changes nothing about which data is served.
`tests/test_governance_organisation_identity.py` covers the model, both modes and
the tenant/organisation separation.

The React path deliberately resolves no organisation: Easy Auth hands this
process a trusted-by-topology header, not a signature-verified directory, and the
SWA principal shape carries no `tid` at all.

### Resource identity — what can be permissioned

`trakt_core/resource.py` names the things an entitlement will later be written
against, and expands each one into a deterministic predicate over existing
canonical fields. It answers *what the resource is*, never *who may reach it* —
nothing in the request path consults it yet, and registering a resource grants
nobody anything.

| Type | Purpose |
|---|---|
| `ResourceRef(tenant_id, kind, resource_id)` | Stable, hashable identity. Canonical string `"{tenant}/{kind}/{id}"`. Kinds: `portfolio`, `source_portfolio`, `spv`, `facility`, `population`. |
| `ResourceRecord` | The catalogue entry: which books, which `spv_id`, which population. |
| `ResolvedResource` | The expansion — `predicate()`, `required_fields`, `to_portfolio_scope()`. |
| `check_attribution(resolved, available_fields)` | Confirms the data can carry the boundary, or raises. |

The model exists because the axes differ in strength, and that difference has to
be explicit rather than assumed:

* `source_portfolio_id` is **mandatory on every onboarded row**, so it can always
  carry a boundary;
* `spv_id` is a **reserved optional** snapshot column (`snapshot.model`), absent
  from the canonical fields registry — an SPV is permissionable only where the
  client supplies it;
* funded / pipeline / forecast are **views** chosen per request (and inferred
  from question wording by `workspace.resolve_active_view`), so population is
  pinned onto the resource as an immutable constraint rather than left to the
  caller.

Two invariants distinguish this from the presentation lens, which must **not** be
reused as an entitlement filter:

* **A resource never widens.** `predicate()` cannot return an empty filter unless
  the record explicitly set `whole_tenant_book`, and a record declaring no
  constraint is rejected at load — so whole-book access is unreachable by
  omission. Compare `resolve_scope`, which answers an unrecognised context with
  Total.
* **Missing attribution refuses.** `check_attribution` raises
  `RESOURCE_NOT_PARTITIONABLE` when the needed column is absent. Compare
  `apply_scope`, which returns the frame unfiltered. A partial-book SPV with no
  attribution is declared `unpartitionable` and always refuses, rather than being
  mapped to its enclosing portfolio.

Config-driven via `config/resources.yaml` (see `config/resources.example.yaml`);
an absent or untrusted file leaves the catalogue **empty**, so every lookup
refuses. Covered by `tests/test_governance_resource_model.py`.

### Portfolio authorisation

`trakt_core.tenancy.authorise_portfolio_access(context, portfolio_id, registry)`
returns an `AuthorisedPortfolio` or raises. Rules:

1. requires the `portfolio:read` scope;
2. a selector naming another configured tenant → `TENANT_MISMATCH`;
3. with an explicit registry, a selector outside the tenant's list/patterns →
   `PORTFOLIO_NOT_AUTHORISED`;
4. selectors and run ids must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`, so a
   value can never traverse a storage path;
5. no portfolio named → the tenant's `default_portfolio`.

Dataset resolution takes the `AuthorisedPortfolio` token, not raw strings, so
there is no code path from a request field to a dataframe that skips this.

**Honest limitation.** With no `config/tenancy.yaml` the deployment is
single-tenant and the tenant owns any well-formed selector inside its own
namespace. Isolation then rests on the dataset root being tenant-scoped
(`…/platform/ERE/…`). Before a second client shares a deployment or a storage
account, add the tenancy config — that is what turns rule 3 on.

### Source approval (no synthetic answers in production)

`trakt_core.policy.evaluate_source_approval` decides on the dataset's
**resolution base**, not its display kind:

| Base | Production | Development / test |
|---|:--:|:--:|
| `platform_canonical` | approved | approved |
| `central_tape` | approved | approved |
| `prepared_explicit` | **refused** | fixture |
| `explicit_csv` | **refused** | fixture |
| `synthetic_demo` | **refused** | fixture |
| `unavailable` | `DATA_SOURCE_UNAVAILABLE` | same |

Deciding on the base is what closes the old gap: pointing `MI_AGENT_DATA_CSV` at
the bundled synthetic demo CSV produced the display kind `explicit_csv`, which
the previous Copilot-only block did not catch.

**What this check does not do.** It classifies by *how the dataset was located*,
not by inspecting the data. A synthetic pack deliberately published as a platform
canonical — which is exactly what `demo_platform` generates — resolves as
`platform_canonical` and is approved. Keeping such a pack out of a deployment is
the storage backend's job, not this policy's (see §6).

The check runs **inside the capability**, so React, Copilot, the deck build and
any future caller are refused identically. There is deliberately **no**
caller-controlled override; `tests/test_governance_source_policy.py::test_no_production_override_flag_exists`
asserts that no such flag exists.

`TRAKT_RUNTIME_MODE` is safe rather than "just another env flag" because:

* unset / empty / unrecognised → `production` (fail closed);
* `validate_runtime_mode()` **raises at import** if a non-production mode is set
  while the Azure markers are present, so it cannot take effect in a deployment;
* `runtime_mode()` independently forces `production` in Azure, so the two cannot
  disagree even if startup validation were bypassed.

### Artefact access

Decks are selected by the authenticated tenant. `client_id` on
`GET /mi/decks/download` is **deprecated**: accepted when it matches the trusted
tenant, refused with `TENANT_MISMATCH` when it names another. React and Copilot
both call `artefacts.get_investor_pack`. The Copilot signed download token is
minted only after that authorisation succeeds.

### Audit metadata

One structured line per governed execution on the `trakt.audit` logger:
capability, request id, correlation id, tenant, actor, actor type, channel,
portfolio, snapshot id, outcome, duration, error code. Never the answer body,
loan rows, tokens, signed URLs or storage paths.

### Azure configuration assumptions

Two controls **cannot** be enforced from this repository. See §5.

---

## 4. Compatibility and deprecations

| Item | Status | Behaviour now | Removal path |
|---|---|---|---|
| React `POST /mi/query` body | **unchanged** | every pre-existing key preserved; additive `governance` block, and `metadata.snapshotId` / `contentHash` / `requestId` | n/a |
| Copilot action responses | **unchanged** | governed refusals add `errorCode` / `retryable` / `category` alongside the documented `ok`/`error` | n/a |
| `client_id` on `/mi/decks/download` | **deprecated** | ignored as an authority; rejected on conflict | remove once the React client stops sending it |
| `MiQueryRequest.client_id` | **deprecated** | portfolio-selector fallback only; rejected on conflict with the trusted tenant | remove when all callers pass `portfolio_id` |
| `mi_agent_api.app._<resolver>` | **moved, re-exported** | now defined in `mi_agent_api.datasets`, re-exported from `app` | remove the re-exports once no caller imports them from `app` |
| `execute_governed_mi_query(req)` | **breaking, internal only** | now `(request, context, dependencies=None)` | done — all call sites updated in this change |
| `/health` `dataSourceInfo` | already removed before this work | withheld because it carried a server-side path | n/a |

The one breaking change is the capability signature. It is internal (no HTTP
contract changed), it was unavoidable — a capability that cannot be given a
trusted context cannot authorise anything — and all three call sites plus their
tests were updated together.

---

## 5. Deployment requirements (not enforceable from code)

| # | Requirement | Why code cannot enforce it | How to verify |
|---|---|---|---|
| D1 | Enable App Service authentication on `trakt-mi-api`, **or** restrict inbound traffic to the Static Web App linked backend | The app cannot know whether its own hostname is publicly reachable | `az webapp auth show -g <rg> -n trakt-mi-api --query enabled` → `true`; or `az webapp config access-restriction show`. Then: `curl -H 'X-MS-CLIENT-PRINCIPAL: <base64 of {"userRoles":["operator"]}>' https://trakt-mi-api.azurewebsites.net/mi/catalogue` must NOT return 200 |
| D2 | Do **not** set `TRAKT_RUNTIME_MODE` in any Azure app setting | The guard raises if it is set, but the setting itself lives in Azure | `az webapp config appsettings list -g <rg> -n trakt-mi-api --query "[?name=='TRAKT_RUNTIME_MODE']"` → empty |
| D3 | Set `TRAKT_COPILOT_DOWNLOAD_SIGNING_KEY` | Multi-worker token validation needs a shared key | app settings contains it; `/v1/copilot/artifacts/latest/investor-deck` link redeems from a second worker |
| D4 | Set `MI_AGENT_CLIENT_ID` | It is the trusted tenant for the deployment | app settings; `/health` reports `tenantId` |

Code-side mitigation for D1: in production, `identity.require_trustworthy_platform_auth()`
refuses every header-authenticated request when the Azure markers are present and
`WEBSITE_AUTH_ENABLED` is unset, unless an operator explicitly declares an
upstream gateway with `MI_AGENT_TRUST_PLATFORM_AUTH=true`. That converts a silent
authentication bypass into a hard, visible failure — but it does not replace D1.

---

## 6. Scope boundaries

### Backend regime processing vs a regulatory-reporting interface

Trakt contains substantial **backend regime-processing capability**: ESMA Annex 2
and Annex 12 projection (`engine/projection_agent`, `engine/gate_4_projection`),
validation (`engine/validation_agent`), XML generation and XSD validation
(`engine/gate_5_delivery`, `engine/delivery_xml_agent`), and the preview policy.
These are preserved unchanged by this work and are executable through the
pipeline and their CLIs.

There is **no regulatory-reporting interface**, and this change did not create
one: no regulatory UI, no regulatory agent tool, no filing or submission
workflow, and no claim of regulatory interface readiness. The central tenant and
source-approval controls apply to the MI and artefact capabilities; extending
them to a regulatory delivery *interface* is future work that starts when such an
interface is specified.

### Landing-page demo

The landing page and demo film serve **committed, pre-generated fixtures**
(`demo-video/public/fixtures/*.json`). Nothing a visitor touches calls the MI API,
and no production endpoint was given a synthetic-data exemption to support them.

State the isolation precisely, because it is **not** the source-approval policy:

* `demo_platform/` is a **capture-time generator**. It deliberately drives the
  real API (`mi_agent_api.app` via `TestClient`) against a synthetic pack in order
  to produce fixtures that are genuine API output. It is never deployed.
* That pack is shaped **exactly like a production platform canonical** — which is
  the point — so its resolution base is `platform_canonical` and the base-level
  approval check does **not** distinguish it from real client data.
* Its isolation therefore rests on the **storage backend**: the pack is only
  reachable with `TRAKT_STORAGE_BACKEND=file` + `TRAKT_LOCAL_BLOB_ROOT`, and in
  Azure `open_storage()` forces blob storage and refuses to fall back to the
  filesystem. A deployed process cannot be pointed at it.

Three tests pin this: `test_production_capability_does_not_depend_on_the_demo_generator`
(dependency runs one way only), `test_landing_page_serves_committed_fixtures_not_a_live_api`,
and `test_demo_pack_is_unreachable_from_an_azure_runtime`.

The generator continues to work unchanged: because the pack resolves as
`platform_canonical` it is approved without needing a non-production runtime mode.

### Known gaps

* **No outbound notification seam** — an agent-to-agent workflow can call in, but
  Trakt cannot call back. Needed before Simulation-B style event workflows.
* **`mi_agent_pptx._api_env` still mutates the environment** in the batch deck
  stage. It is not on a per-request path; removing it means threading an explicit
  dataset selector through `data_source`, which changes resolution for every
  caller at once and is better done as its own change.
* **Single-tenant namespace default** — see §3.
