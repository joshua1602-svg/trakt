# MI Query as an A2A agent — architecture and implementation-readiness review

**Status: MINOR_ARCHITECTURAL_REMEDIATION**

Reviewed at `93d60b2`. No production code was changed to produce this review.

---

## Summary

The hard part is already done and done well. `mi_agent_api.mi_service.execute_governed_mi_query`
is a genuine single governed capability that React and Copilot both call, with a
channel-neutral request model, a trusted-identity model that request bodies cannot
influence, governance ordered before data access, and a structural-and-numeric parity
suite already asserting the two channels agree. Exposing it to a third caller needs an
adapter, not an engine.

Three things stand between here and a production MI Query A2A agent. None requires a
second calculation path; one is a genuine design decision rather than wiring.

1. **The A2A framework has no transport.** `trakt_a2a` is complete as a protocol
   implementation and is reachable only in-process. There is no HTTP route, no
   `/.well-known/agent-card.json`, and no code path that turns a validated Entra token
   into a `CallerIdentity`. This blocks *any* A2A agent, not just MI Query.

2. **MI Query and the agent surface authorise differently.** MI authorises portfolios
   against the tenant registry; machine identities carry organisation-scoped resource
   grants that MI never consults. Exposing MI Query to agent identities as it stands
   would weaken the boundary the agent tool API currently holds.

3. **The repository contains a documented decision against exactly what is being asked
   for.** `agent_api.py` excludes the natural-language MI interpreter from the machine
   surface, with a stated reason. That decision may be right or wrong, but it should be
   overturned deliberately rather than by accident.

A correction to the brief's premise: **there is no Teams MI integration.** The Teams
router is registration and proactive notification only and carries no portfolio data.

---

## A. Current-state architecture

### A.1 The MI Query pathway (actual)

```
POST /mi/query                        app.py:1722   channel=react      Easy Auth principal
POST /v1/copilot/mi/query   copilot_actions.py:359  channel=copilot    Entra delegated token
in-process (simulation, jobs)                       channel=internal   trusted invocation
        │
        │   each builds an ExecutionContext from a VERIFIED identity, then calls:
        ▼
mi_agent_api.mi_service.execute_governed_mi_query(MiQueryRequest, ExecutionContext, deps)
        capability id: "mi.question.answer"
        │
        ├── 1. context.require_scope(SCOPE_MI_QUERY)
        ├── 2. client_id / tenant disagreement → TENANT_MISMATCH
        ├── 3. trakt_core.tenancy.authorise_portfolio_access(context, portfolio_id, registry)
        │        └── require_scope(SCOPE_PORTFOLIO_READ); tenant from context, never body
        ├── 4. deps.datasets.describe_active_dataset()
        │      trakt_core.policy.evaluate_source_approval(...)
        ├── 5. _run_analysis  →  mi_agent.mi_agent_workflow.run_mi_agent_query
        │        parse once → recogniser registry → routed capability
        │        or deterministic point-in-time executor
        │        (LLM is parser-only and optional; parser_mode
        │         "deterministic" | "llm"; the deterministic stack is the control layer)
        └── 6. GovernedResult[dict] + trakt_core.audit.emit_audit_event
        │
        ▼
presenters.to_react_payload(result)          → React envelope + additive `governance`
copilot_actions inline shaping               → CopilotMiAnswer + supporting values
```

**Properties that hold today, verified in source:**

| Property | Where |
|---|---|
| No web framework below the adapter | `mi_service` imports no FastAPI; asserted by `tests/test_governance_dependency_direction.py` |
| Tenant is never request data | `ExecutionContext` docstring and `authorise_portfolio_access` |
| Governance before data | steps 1–4 precede any dataframe |
| Analytical payload identical across channels | `mi_agent_api/tests/test_channel_parity.py` over the golden-question library |
| Governed failures cannot become narrative | `GovernedResult` with typed `TraktError`, never raises for analytical fault |

**Answer to the brief's conditional question:** React and Copilot *do* share a genuinely
common governed pathway. This is not an architectural defect and needs no remediation.

### A.2 The single reusable service boundary

```python
mi_agent_api.mi_service.execute_governed_mi_query(
    request: MiQueryRequest,          # untrusted caller input
    context: ExecutionContext,        # trusted identity
    dependencies: CapabilityDependencies | None = None,
) -> GovernedResult[dict]
```

This is the boundary A2A should invoke. Nothing above it is reusable (it is HTTP and
presentation); nothing below it should be reached directly.

### A.3 The two other machine surfaces that already exist

**`/v1/agent/tools` — the external agent tool API** (`agent_api.py`, off by default behind
`TRAKT_AGENT_API_ENABLED`). Entra client-credentials token, `Trakt.Agent` app role,
`agent_auth` reusing the Copilot directory allow-list and JWKS. 27 typed tools over
`trakt_tools.execute_governed_tool`. Returns the full `GovernedResult.to_dict()`.

**`trakt_tools/mcp_server.py` — an MCP server** over the same tool registry,
`MCP_PROTOCOL_VERSION = "2025-06-18"`, `initialize` / `tools/list` / `tools/call`.

Both are narrower than MI Query by design and both already satisfy "same governed
execution, different transport".

### A.4 The Securitisation Readiness A2A implementation

| Component | File | Reusable for MI Query? |
|---|---|---|
| Agent Card builder, A2A v1.0, `supportedInterfaces` with per-interface `protocolVersion` | `card.py` | **Pattern yes, content no** — hard-codes one `SKILL_ID`, readiness description and limits |
| JSON-RPC 2.0 dispatch, `message/send`, `tasks/get`, bounded errors | `server.py:163` | **Yes** — protocol-generic |
| `TaskStore`, lifecycle `submitted → working → completed/failed/rejected`, terminal-state guard | `tasks.py` | **Yes** |
| `CallerIdentity` — `agent_id` and `organisation_id` separate | `server.py:100` | **Yes** |
| Two-step authorisation (authenticate ≠ authorise) | `identity.py` | **Yes** |
| `caller_from_principal` reusing `context_from_agent_principal` | `identity.py:42` | **Yes** |
| `OUT_OF_SCOPE` keyword refusals | `server.py:63` | **Pattern yes, content no** — readiness-specific |
| `readiness_artifact`, `_finding`, `_answer_from_assessment` | `server.py:343` | **No** — specialist output |
| Assessor / session injection | `server.py:132` | **Yes** — the extension point |
| Correlation: `context.with_correlation(task.id)` | `run_a2a_eval.py:91` | **Yes** |
| Test coverage | `tests/test_a2a_delegation.py` (39), `test_a2a_governance_and_audit.py` (14) | Harness reusable |

**What is missing from it entirely:**

- No HTTP transport. `grep` for `trakt_a2a` outside its own package returns
  `scripts/run_a2a_eval.py` and tests only. `SecuritisationReadinessA2AServer.handle()`
  is an in-process method.
- No `/.well-known/agent-card.json` route.
- No token→`CallerIdentity` binding in a request path. `caller_from_principal` exists but
  nothing calls it outside tests.
- No cancellation (`tasks/cancel` is not implemented; `message/stream` is not implemented
  and the card correctly declares `streaming: false`).
- No `pushNotifications`.

---

## B. Proposed target architecture

```
   React        Copilot        M365 Copilot        Copilot Studio /      MCP client
   (Easy Auth)  (delegated)    declarative agent   Finova ent. agent
      │            │            │                      │                    │
      │            │            │  API plugin          │  A2A               │  MCP
      │            │            │  (OpenAPI)           │  JSON-RPC          │
      ▼            ▼            ▼                      ▼                    ▼
 ┌─────────┐ ┌──────────┐ ┌──────────────────────────────────────────────────────┐
 │/mi/query│ │/v1/copilot│ │        Trakt machine surfaces (all adapters)         │
 │         │ │/mi/query  │ │  /v1/agent/tools    /a2a (new)    MCP server         │
 └────┬────┘ └─────┬────┘ └───────┬───────────────────┬─────────────────┬────────┘
      │            │              │                   │                 │
      │            │              │                   │                 │
      └────────────┴──────────────┴─────────┬─────────┴─────────────────┘
                                            │
                          ┌─────────────────┴──────────────────┐
                          ▼                                    ▼
        mi_service.execute_governed_mi_query      trakt_tools.execute_governed_tool
              "mi.question.answer"                     27 typed tools
                          │                                    │
                          └─────────────────┬──────────────────┘
                                            ▼
                       ExecutionContext · scopes · entitlements · policy
                                            ▼
                        Deterministic analytics · canonical portfolio
                                            ▼
                              GovernedResult + audit event
```

The specialist agents (Securitisation Readiness, future Acquisition DD) sit *above* the
governed capability layer as orchestrators that consume tools — they are not a peer of it.
The brief's proposed diagram places them alongside MI Query Agent; in this repository the
relationship is:

```
Securitisation Readiness Agent  ──consumes──►  trakt_tools (27 tools)  ──►  governed layer
MI Query A2A adapter            ──consumes──►  mi_service              ──►  governed layer
```

That is a real distinction, not pedantry: the readiness agent is an LLM loop that decides
what to call; the MI Query adapter must not be.

---

## C. Reuse assessment — what A2A should call

| Concern | Reuse | Do not build |
|---|---|---|
| MI execution | `mi_service.execute_governed_mi_query` | any second engine, any "simplified" MI service |
| Request model | `mi_service.MiQueryRequest` | an A2A-specific request type |
| Identity | `mi_agent_api.identity.context_from_agent_principal` | a new identity model |
| Token validation | `mi_agent_api.agent_auth` (Entra, `Trakt.Agent`, JWKS, directory allow-list) | a second validator |
| Entitlement | `trakt_core.entitlement.authorise_resource_access` | a bespoke check |
| Result envelope | `trakt_core.envelope.GovernedResult` | a new envelope |
| Structured response | `GovernedResult.to_dict()` + the existing analytical payload | new schemas |
| Human-readable answer | `envelope["answer"]` — the engine already produces it | LLM re-summarisation |
| Audit | `trakt_core.audit.emit_audit_event` (already inside the capability) | A2A-specific audit |
| A2A protocol | `trakt_a2a.tasks`, JSON-RPC dispatch, `CallerIdentity` | a second A2A stack |
| Parity testing | `mi_agent_api/tests/test_channel_parity.py` | a new parity harness |

---

## D. Gap analysis

### D.1 Blocking — A2A has no transport

**Severity: blocking. Effort: small. Risk: low.**

Everything needed exists; nothing is joined up. Required:

- a router mounting `GET /.well-known/agent-card.json`
- a router mounting `POST /a2a` (JSON-RPC 2.0)
- a dependency that validates the Entra token via `agent_auth`, resolves the organisation,
  and builds `CallerIdentity` via `trakt_a2a.identity.caller_from_principal`
- `enabled()` gating, matching `agent_api.enabled()` and `teams_bot.enabled()`

This is a prerequisite for the *existing* Securitisation Readiness agent too. It has never
been reachable from outside the process.

### D.2 Material — the authorisation models do not meet

**Severity: material. Effort: small. Risk: medium (touches the shared capability).**

Two different functions authorise portfolio access:

```
MI Query    trakt_core.tenancy.authorise_portfolio_access(context, portfolio_id, registry)
              → require_scope(portfolio:read)
              → registry.resolve(context.tenant_id)
              → record.allows_selector(selector)
              ✗ never reads context.permitted_resources

Tools/A2A   trakt_core.entitlement.authorise_resource_access(...)
              → scope gate, then per-resource organisation grant
              → RESOURCE_NOT_AUTHORISED
```

A machine principal gets `tenant_id = default_tenant_id()` (deployment configuration —
deployment-per-tenant is the production model) and **scopes derived from its organisation's
grants** (`identity.scopes_from_entitlements`). So an external organisation granted
`mi:query` + `portfolio:read` on *one* portfolio would pass MI's scope gate, and
`authorise_portfolio_access` would then authorise any selector the **deployment tenant** is
allowed — because it never consults that organisation's resource grants.

Because the deployment is single-tenant, this is not a cross-tenant leak. It is a
**portfolio-level authorisation gap within a tenant**, and it becomes live the moment MI
Query is exposed to an agent identity. It is precisely the "divergent permission logic" the
brief prohibits, and it is why the agent tool API's decision to route machines through
`execute_governed_tool` — which *does* check entitlements — has been holding the line.

`trakt_a2a.identity.authorise()` is explicitly documented as *not* the security boundary,
and is right to say so: for the readiness agent, `execute_governed_tool` is the enforcement.
For MI Query there would be no equivalent enforcement underneath.

**Recommended fix:** an additive check inside `mi_service`, gated on
`context.actor_type == ACTOR_SERVICE`, that resolves the requested portfolio through
`authorise_resource_access` in addition to `authorise_portfolio_access`. Human paths are
bit-identical; machine paths gain the check the tool path already has. Roughly 15 lines in
one function, with the parity suite as the regression net.

### D.3 Design decision — natural language on the machine surface

**Severity: decision required, not a defect.**

`agent_api.py:18–22` states:

> **The natural-language MI interpreter.** `POST /mi/query` takes prose and runs Trakt's own
> LLM over it. Routing an external agent through that would put two probabilistic hops in
> one call — the agent choosing words and Trakt guessing what they meant. An agent calls a
> typed tool with typed arguments; choosing *which* tool is the only inference in the loop.

The brief asks for exactly the surface this excludes. The argument is not wrong, but it is
weaker than it looks, for two reasons found in the code:

- The MI parser is **not necessarily an LLM**: `parser_mode` is
  `"deterministic" | "llm"` and the LLM path is optional and repairable. The
  "two probabilistic hops" concern is configuration-dependent.
- The interpreted result is **disclosed, not hidden**: the envelope carries `interpreted`,
  `spec`, `validation.resolved_fields` and `assumptions`. A calling agent can verify what
  Trakt understood before trusting the number — which is a stronger position than a typed
  tool call, not a weaker one.

**Recommendation:** overturn it deliberately, and mitigate rather than ignore it — return
`interpreted` and `spec` in the A2A artifact so the caller can detect a misread, and refuse
rather than guess on low-confidence parses (the engine already reports unmet/ambiguous
outcomes as `ok: false`).

### D.4 Minor gaps

| Gap | Severity | Note |
|---|---|---|
| `server.py` hard-codes one `SKILL_ID` and a readiness artifact | minor | needs a skill→handler map; the assessor injection already shows the shape |
| No `tasks/cancel` | minor | MI Query is seconds; matters less than for readiness |
| No streaming | minor | correctly declared `streaming: false` |
| `OUT_OF_SCOPE` is readiness-specific | minor | MI Query needs its own, or none |
| No rate limiting on any agent surface | **should fix** | see F |
| `TaskStore` is in-memory | minor for MI Query | material for multi-instance readiness |

---

## E. Finova / Microsoft interoperability assessment

Sources are listed at the end. Claims are labelled.

### E.1 Is this architecture market-standard?

**Yes — confirmed.** A2A reached **v1.0 in April 2026** and is governed by the Linux
Foundation, with **150+ organisations** participating and stated production use in financial
services. v1.0.1 (May 2026) added an extension mechanism. Trakt's card is already written
to v1.0 with per-interface `protocolVersion`, which is where v1.0 moved it.

### E.2 Could a Finova enterprise agent reasonably consume it?

**Technically yes — confirmed for the protocol, inference for the vendor.**

*Confirmed:* Finova partnered with Covecta in January 2026 to embed agentic AI across its
lending and broker platforms; Broker Assist launched March 2026. Covecta builds configurable
agents for banks, building societies and specialist lenders, explicitly including portfolio
monitoring and due diligence.

*Reasonable inference:* an agent platform of that description consuming an external
specialist agent over A2A is the intended use of the protocol, and Trakt's card is
standards-conformant.

*Speculation, and labelled as such:* there is **no public evidence that Finova or Covecta
implement A2A today.** Do not plan on the assumption that they do. The commercially safer
position is that Trakt supports A2A *and* REST/OpenAPI *and* MCP, so the integration
mechanism is Finova's choice rather than a precondition.

### E.3 Could a Finova Microsoft Copilot / Copilot Studio agent consume it?

**Copilot Studio: yes — confirmed.** A2A connections reached **general availability in
Copilot Studio in April 2026**, with agent-to-agent communication graduating to GA in the
May update. A Copilot Studio agent can delegate to first-, second- or third-party agents
over A2A, and agents built with the Microsoft Agent Framework SDK are wire-compatible.

**Microsoft 365 Copilot declarative agents: no — use a different mechanism.** M365 Copilot
extensibility is **API plugins (OpenAPI) and MCP**, not A2A. A declarative agent grounds and
calls tools; it does not delegate tasks to a remote agent over A2A.

### E.4 Is A2A the right protocol for each case?

| Consumer | Right mechanism | Trakt status |
|---|---|---|
| Finova / Covecta enterprise agent | **A2A** (or REST if they prefer) | A2A framework exists, unmounted; REST exists |
| Copilot Studio agent | **A2A** — GA since April 2026 | same |
| M365 Copilot declarative agent | **API plugin (OpenAPI)** or **MCP** | `/v1/agent/tools` exists; MCP server exists; no OpenAPI doc published for the agent API |
| Power Automate / Logic Apps | **Custom connector** over the REST surface | REST exists |
| Azure API Management fronting any of these | orthogonal — a deployment concern | not present |

**Do not force A2A everywhere.** The correct reading of the current market is that A2A is
for *agent delegates task to agent*, and OpenAPI/MCP is for *agent calls a tool*. MI Query
is genuinely the former when a peer agent hands over a business question, and genuinely the
latter when Copilot grounds an answer. Trakt should support both, and it very nearly
already does.

### E.5 What else should Trakt support?

1. **Publish an OpenAPI description of `/v1/agent/tools`.** This is the single highest
   commercial-leverage item in this review and is close to free: it makes Trakt consumable
   by M365 Copilot declarative agents, Copilot Studio custom connectors, Power Automate and
   Azure APIM without any new capability.
2. **Signed Agent Cards.** `agent_card()` already accepts `signatures`; nothing produces
   them. An enterprise counterparty verifying the card is a credible near-term ask.
3. Keep the MCP server — it is the M365/Anthropic-side lingua franca for tool calling.

---

## F. Security assessment

| Control | Status | Note |
|---|---|---|
| Entra / OAuth2 client credentials | **present** | `agent_auth`, JWKS, issuer + audience validation |
| Required app role (`Trakt.Agent`) | **present** | defaults to requiring it rather than empty |
| Directory allow-list | **present** | shared with Copilot, pinned by test against silent forking |
| Fail-closed auth mode | **present** | `disabled` mode refused outright in production runtime |
| Service-to-service authentication | **present** | app-only token, `actor_type=service` |
| Delegated user authorisation (on-behalf-of) | **absent** | documented as a later mechanism; acceptable for machine-to-machine, a gap if a named end user must be carried |
| Tenant isolation | **present** | deployment-per-tenant; tenant never from request body |
| Portfolio-level permission for machines on MI | **GAP — see D.2** | must be closed before exposure |
| Skill-level permission | **partial** | card declares `securityRequirements` per skill; not enforced per skill in `server.py` |
| Identity never from request body | **present** | enforced in `mi_service`, `trakt_tools.mcp`, `trakt_a2a` |
| Enumeration-safe refusals | **present** | unauthorised and nonexistent refused identically |
| Audit logging | **present** | `emit_audit_event` inside the capability, with correlation id |
| Correlation propagation | **present** | `X-Request-Id` / `X-Correlation-Id` honoured |
| Error containment | **present** | bounded messages; no stack traces across the boundary |
| Replay protection | **absent** | no nonce/jti cache; token lifetime is the only bound |
| Rate limiting | **absent** | **should fix** — an A2A task can start an expensive LLM run |
| Prompt injection | **partially mitigated** | MI Query's exposure is a misread question, not tool abuse, and `interpreted`/`spec` disclose the reading. The readiness agent has the larger surface. |
| Output filtering / data leakage | **present by construction** | the capability returns only what the governed engine produced |
| Secrets | **not in scope of this review** | no secret values were read or handled |

**Blockers before exposure: D.2 (portfolio-level authorisation for machine callers).**
**Strongly recommended before exposure: rate limiting on the A2A surface.**

---

## G. Proposed Agent Card and skills — illustrative

**Recommendation: Option A — a separate Agent Card, served by a shared A2A framework.**

Reasons, in order of weight:

1. **The published limits are incompatible.** The readiness card states "does not forecast",
   "does not run scenarios". MI Query legitimately answers forecast and scenario questions —
   `_run_analysis` routes to forecast and scenario capabilities. One card cannot carry both
   sets of limits without lying about one of them.
2. **The security postures differ.** `mi:query` is documented in `context.py` as "a much
   wider grant" than the risk read. A counterparty may reasonably be given a readiness
   assessment and not the free-form MI surface. Separate cards make that grantable.
3. **The task profiles differ.** Readiness is minutes and genuinely task-based; MI Query is
   sub-second-to-seconds and is naturally request/response. Card-level `capabilities` cannot
   express both well.
4. Option B (one card, more skills) is cheaper today and wrong at the second specialist.

```jsonc
{
  "name": "Trakt Portfolio Intelligence Agent",
  "description": "Answers governed questions about a loan portfolio. Every figure is
                  computed by Trakt's deterministic engine from the canonical portfolio,
                  never estimated by a language model, and returned with the scope,
                  as-at date and provenance behind it.",
  "version": "1.0.0",
  "provider": { "organization": "Trakt", "url": "https://trakt.example" },
  "supportedInterfaces": [
    { "url": "https://…/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0" }
  ],
  "capabilities": { "streaming": false, "pushNotifications": false,
                    "stateTransitionHistory": true },
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["application/json", "text/plain"],
  "securitySchemes": { "enterprise_agent_oauth": { /* as the readiness card */ } },
  "securityRequirements": [{ "enterprise_agent_oauth": ["Trakt.Agent"] }],
  "skills": [{
    "id": "portfolio_question",
    "name": "Governed portfolio question",
    "description": "Ask one question about a governed portfolio and receive the answer
                    with its computed values, the portfolios it covers, the as-at date
                    and the provenance of every figure.",
    "tags": ["portfolio", "credit", "mi", "reporting", "risk"],
    "examples": [
      "What is total funded balance by SPV?",
      "Compare current LTV across the acquired and originated books.",
      "Which regions carry the highest exposure?",
      "Which concentration limits are currently breached?",
      "How has portfolio composition changed since last month?"
    ],
    "inputModes": ["text/plain"],
    "outputModes": ["application/json"]
  }]
}
```

**One skill, not four.** `portfolio_compare`, `portfolio_risk_analysis` and
`portfolio_drilldown` are not separate capabilities in this codebase — they are routes the
recogniser registry already selects *inside* `_run_analysis` from the question itself.
Publishing four skills would advertise a distinction Trakt does not implement and force the
caller to make a routing decision Trakt makes better.

**Stated limits** (the honest ones for this agent):

- Answers only from the governed canonical portfolio; declines rather than estimates.
- Does not issue credit ratings, PD, LGD or expected-loss estimates.
- Does not perform multi-step investigation — for that, delegate to the Securitisation
  Readiness Agent.

**Artifact media type:** `application/vnd.trakt.portfolio-answer+json`.

**Response contract** — both halves, from data the engine already produces, nothing new
invented:

```jsonc
{
  "skill": "portfolio_question",
  "question": "…",                      // as asked
  "answer": "…",                        // envelope.answer — engine-produced prose
  "interpreted": "…",                   // envelope.interpreted — what Trakt understood
  "spec": { … },                        // envelope.spec — metric, dimensions, filters
  "values": [ … ],                      // envelope.artifacts — the computed rows
  "scope": { … },                       // GovernedResult.scope — portfolios in/used
  "asAt": "…",                          // snapshot.reporting_date
  "snapshot": { "id": "…", "contentHash": "…", "rowCount": … },
  "provenance": { "sourceNotes": [ … ], "reconciliation": { … } },
  "validation": { … },                  // envelope.validation
  "assumptions": [ … ], "warnings": [ … ],
  "capability": "mi.question.answer",
  "requestId": "…", "correlationId": "…"
}
```

Every field above already exists on `GovernedResult` or in the analytical envelope. The
adapter renames and selects; it must not compute.

---

## H. Minimal implementation plan

Smallest change set that exposes the existing capability. Additive throughout; no
refactor of `mi_service`'s existing behaviour.

| # | Change | File | Size |
|---|---|---|---|
| 1 | Entitlement check for machine callers on the MI path, gated on `actor_type == service` | `mi_agent_api/mi_service.py` | ~15 lines |
| 2 | Generalise the A2A server: skill registry `{skill_id: handler}`, per-skill card assembly | `trakt_a2a/server.py`, `card.py` | moderate |
| 3 | MI Query skill handler — builds `MiQueryRequest`, calls the capability, maps `GovernedResult` to the artifact | `trakt_a2a/skills/portfolio_question.py` (new) | ~120 lines, no analytics |
| 4 | HTTP transport: agent-card route, JSON-RPC route, token→`CallerIdentity` dependency, `enabled()` gate | `mi_agent_api/a2a_api.py` (new) + mount in `app.py` | ~120 lines |
| 5 | Tests — see J | `tests/` | — |

Not in scope: streaming, cancellation, push notifications, persistent task store, OpenAPI
publication (recommended separately), signed cards.

---

## I. Recommendation

### MINOR_ARCHITECTURAL_REMEDIATION

**Not `READY_TO_IMPLEMENT`**, because of D.2: exposing MI Query to agent identities today
would give an external organisation portfolio access its grants do not confer. That is a
real authorisation gap, it is in the shared capability rather than the adapter, and the
brief's own non-negotiable #7 forbids shipping over it.

**Not `MATERIAL_ARCHITECTURAL_REMEDIATION`**, because nothing structural is wrong. There is
one MI engine, one governed capability, one identity model, one envelope, one audit path,
and a parity suite already proving two channels agree. The A2A framework exists and is
tested. Every gap is additive to fill and none requires touching the analytics.

**Not `A2A_NOT_CURRENTLY_JUSTIFIED`**, because the market evidence is now unambiguous — A2A
v1.0 under the Linux Foundation, GA in Copilot Studio since April 2026 — and because the
strategic case (§7 of the brief) is sound: Trakt already separates capability from transport
cleanly enough that a fourth surface is an adapter. That is the proof of the positioning,
and it is worth more than the protocol itself.

**On the strategic question:** A2A is not defensible IP and this review does not treat it as
such. What the review does show is that Trakt's defensibility — the governed portfolio
model, deterministic calculation, provenance, entitlements and audit — is *already*
factored so that any transport reaches it identically. A competitor could implement A2A in a
week; what they could not do is answer the same question the same way through four surfaces
and prove it with a parity suite. The interoperability story is credible precisely because
the boundary was drawn before anyone asked for it.

**Sequencing recommendation:** do D.2 and the HTTP transport (H.1 and H.4) first, and mount
the **existing Securitisation Readiness agent** over them. That delivers a working, reachable
A2A endpoint, validates the transport against an agent that is already fully tested, and
de-risks the MI Query skill to a pure adapter exercise. Publishing the OpenAPI description
of `/v1/agent/tools` (E.5.1) can proceed in parallel and may deliver more commercial value
sooner than A2A does.

---

## J. Execution prompt

See `docs/mi_query_a2a_execution_prompt.md`.

---

## Sources

- [Linux Foundation — A2A Protocol Project launch](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents)
- [Linux Foundation — A2A surpasses 150 organizations, production use in first year](https://www.linuxfoundation.org/press/a2a-protocol-surpasses-150-organizations-lands-in-major-cloud-platforms-and-sees-enterprise-production-use-in-first-year)
- [Agent2Agent — Wikipedia](https://en.wikipedia.org/wiki/Agent2Agent)
- [Microsoft Learn — Connect to an agent over the Agent2Agent (A2A) protocol (Copilot Studio)](https://learn.microsoft.com/en-us/microsoft-copilot-studio/add-agent-agent-to-agent)
- [Microsoft Copilot Blog — Updates to multi-agent systems](https://www.microsoft.com/en-us/microsoft-copilot/blog/copilot-studio/new-and-improved-multi-agent-orchestration-connected-experiences-and-faster-prompt-iteration/)
- [Microsoft Learn — Build cross-platform multi-agent solutions using A2A in Copilot Studio](https://learn.microsoft.com/en-us/training/modules/build-cross-platform-multi-agent-solutions-agent2agent-copilot-studio/)
- [Microsoft Learn — Plugins for Microsoft 365 Copilot](https://learn.microsoft.com/en-us/microsoft-365/copilot/extensibility/overview-plugins)
- [Microsoft Learn — Agents for Microsoft 365 Copilot](https://learn.microsoft.com/en-us/microsoft-365/copilot/extensibility/agents-overview)
- [Finova — Partnership with Covecta](https://www.finova.tech/blog/finova-partners-with-covecta-to-bring-agentic-ai-capabilities-to-the-uk-lending-market)
- [Covecta — Finova launches Broker Assist AI agent](https://www.covecta.io/post/finova-launches-broker-assist-ai-agent)
- [Mortgage Solutions — Finova embeds AI agent into platform](https://www.mortgagesolutions.co.uk/news/2026/03/10/finova-embeds-ai-agent-into-platform-to-answer-brokers-lending-queries/)
