# The Trakt agent tool API

How an external AI agent — Claude, an OpenAI-based agent, Microsoft Copilot, or a
client's own — authenticates to Trakt, discovers what it may do, and calls a
governed credit capability.

Audience: engineers adding a tool, and reviewers checking the security model. It
documents the code as it is, not an intended end state.

> **Status.** Sprint 1 of the A2A programme (see
> [`a2a_architecture_readiness_review.md`](a2a_architecture_readiness_review.md)).
> One tool is registered — `evaluate_covenants` — deliberately: the sprint's
> question is whether the *architecture* works end to end, and one tool answers
> it as well as fifteen would while failing faster if it does not.

---

## 1. The governed call flow

```
  Claude          OpenAI agent      Microsoft Copilot      client's own agent
       │                │                    │                      │
       └────────────────┴─────────┬──────────┴──────────────────────┘
                                  │  HTTPS + Entra client-credentials bearer
                                  ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ TRANSPORT + IDENTITY        mi_agent_api/agent_auth.py                     │
 │   validate token signature / issuer / audience / expiry against ONE        │
 │   allow-listed directory (JWKS), then require the Trakt.Agent app role     │
 │                             mi_agent_api/identity.py                       │
 │   → ExecutionContext(tenant, actor=service, channel=enterprise_agent,      │
 │                      organisation, entitlements, scopes DERIVED)           │
 └───────────────────────────────┬───────────────────────────────────────────┘
                                 ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ ROUTE                       mi_agent_api/agent_api.py                      │
 │   GET  /v1/agent/tools               the catalogue THIS caller may use     │
 │   POST /v1/agent/tools/{tool_name}   typed JSON arguments                  │
 │   No business logic. No schema decisions. No permission decisions.         │
 └───────────────────────────────┬───────────────────────────────────────────┘
                                 ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ GOVERNED EXECUTION          trakt_tools/execution.py                       │
 │   1 tool exists                    registry                                │
 │   2 arguments match the schema     trakt_tools.schema                      │
 │   3 caller holds the capability    ExecutionContext.require_scope          │
 │   4 organisation entitled to the   trakt_core.entitlement                  │
 │     NAMED RESOURCE                   .authorise_resource_access            │
 │   5 dataset approved to answer     trakt_core.policy                       │
 │   ── nothing above touches data; every check precedes the first read ──    │
 │   6 EXISTING deterministic handler                                         │
 │   7 GovernedResult                 trakt_core.envelope                     │
 │   8 audit                          trakt_core.audit                        │
 └───────────────────────────────┬───────────────────────────────────────────┘
                                 ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ DOMAIN — UNCHANGED                                                         │
 │   mi_agent_api/concentration_tests_api.compute_concentration_tests         │
 │     ↳ the SAME function GET /mi/concentration-tests calls                  │
 │   mi_agent/concentration_tests/{evaluation,metrics,library}                │
 └───────────────────────────────────────────────────────────────────────────┘
```

### Package boundaries

| Package | Rule |
|---|---|
| `trakt_tools/` | Tool declarations and the governed executor. No FastAPI. The registry imports nothing heavy at all — asserted by a subprocess test. |
| `trakt_tools/handlers/` | Thin wrappers over existing implementations. **No tool computes anything.** |
| `mi_agent_api/agent_auth.py`, `agent_api.py` | The only agent modules importing FastAPI. |
| `mi_agent/`, `analytics_lib/`, `engine/` | Domain. Untouched by this work. |

---

## 2. The three rules, and the tests that hold them

**No tool computes anything.** Every handler wraps an implementation the UI
already calls. `evaluate_covenants` calls
`concentration_tests_api.compute_concentration_tests` — the same function
`GET /mi/concentration-tests` calls — and then *re-keys* the result into the
tool's published schema. Re-keying is allowed; rounding, defaulting a missing
number to zero, or re-deriving one is not.

> `test_the_covenant_tool_calls_the_same_implementation_the_ui_calls`
> `test_the_projection_carries_every_value_verbatim`

Why it matters: an agent and the workspace reporting different limit usage for
the same book is not a display bug once the agent's number is the one a
counterparty sees.

**Governance runs before data.** Steps 1–5 touch no dataframe. This is asserted
by *counting* calls to the dataset resolver on every refusal path, not by reading
the source.

> `test_nothing_reads_data_before_authorisation_succeeds`

**Failure is always an envelope.** `execute_governed_tool` never raises across
the boundary. A caller always receives a `GovernedResult` with a typed error, and
every path — including every refusal — emits an audit event first.

> `test_execute_governed_tool_never_raises_across_the_boundary`
> `test_every_path_emits_exactly_one_audit_event`

---

## 3. Identity: a machine is not a human

| | Human (React / Copilot) | Machine (agent API) |
|---|---|---|
| Token | Easy Auth header, or delegated Entra | Entra **client credentials** |
| `actor_type` | `user` | `service` |
| `channel` | `react` / `copilot` | `enterprise_agent` |
| Scopes | `DEFAULT_MI_SCOPES` — a fixed set for a role | **Derived from its grants** |
| Organisation | resolved from a verified `tid`, optional | resolved from a verified `tid`, **required** |
| Principal registry | consulted (per-individual kill switch) | **not consulted** |

Three of those deserve their reasoning stated.

**Scopes are derived, not defaulted.** `identity.scopes_from_entitlements` returns
the union of the capabilities the organisation's grants confer. A machine never
inherits a human's scope set, and there is no second place to write a machine's
permissions down. The union does not widen anything: `scopes` only answers *may
this caller attempt this verb at all*, and `authorise_resource_access` then
answers *may it against this resource*. An agent granted `risk:read` on Portfolio
A and nothing on Portfolio B passes the scope gate for both and is refused at the
resource gate for B — which is the right place, and the one that returns
`RESOURCE_NOT_AUTHORISED` rather than leaking that B exists.

**An unregistered directory is refused.** The human channels fall back to
*compatibility mode* when no organisation registry exists, because they authorise
through the tenant/portfolio path. The agent surface must not: every tool call
authorises through entitlements, and entitlements hang off an organisation.
Without one there is nothing to check, so the honest answer is to refuse rather
than serve an unattributed machine.

**The principal registry is not consulted.** It binds *individuals* by
`(microsoft_tenant_id, microsoft_object_id)`. A service principal's `oid` is an
application. Consulting the registry would either refuse every agent on a
principal-gated directory or register machines as staff. An agent acting *on
behalf of* a named user is a separate mechanism — an `on_behalf_of` claim — and
is not built.

### Configuration

| Variable | Meaning |
|---|---|
| `TRAKT_AGENT_API_ENABLED` | Mounts the router. **Off by default** — exposing a machine-callable surface is a deployment decision. |
| `TRAKT_AGENT_AUTH_MODE` | `entra` (default, fail closed) \| `disabled` (local only; refused outright in production) |
| `TRAKT_AGENT_ENTRA_AUDIENCE` | Accepted audience(s). Required in `entra` mode. |
| `TRAKT_AGENT_REQUIRED_ROLE` | App role a token must carry. Defaults to `Trakt.Agent` — an agent surface should need a deliberate role assignment, not admit any valid token from an accepted directory. |
| `TRAKT_AGENT_DEV_DIRECTORY` | `disabled` mode only. Even locally the caller is never anonymous. |

`auth.auth_guard` — the global Easy Auth dependency on the app — exempts
`/v1/agent`, exactly as it already exempts `/v1/copilot` and `/v1/teams`. A
machine identity carries no `X-MS-CLIENT-PRINCIPAL` header, so without the
exemption every agent call is refused 401 by the wrong guard before `agent_auth`
runs. That was a real defect during this sprint: the router tested in isolation
passed, because the global dependency lives on the *app*, not on the router.
`test_the_easy_auth_guard_exempts_the_agent_prefix` covers it, and also asserts
the exemption did not widen to anything else.

The directory allow-list is shared with `copilot_auth`: it is the union of
`TRAKT_COPILOT_ENTRA_TENANT_ID` and every enabled organisation's directories, so
registering an organisation is enough and no GUID has to be duplicated into an
app setting. `agent_auth` reuses `copilot_auth`'s allow-list and key-selection
helpers rather than restating them; a test pins those names so a rename breaks
loudly instead of silently forking the allow-list.

---

## 4. Authorisation is per resource

An agent names a **resource** — `{tenant}/{kind}/{resource_id}` — never a dataset,
a file or a portfolio selector. The server produces the predicate.

```
GET /v1/agent/tools
{
  "tools": [ { "name": "evaluate_covenants", "version": "1.0.0", ... } ],
  "resources": { "risk:read": ["ERE/source_portfolio/direct_001"] },
  "organisation_id": "a2a_test_agent",
  "tenant_id": "ERE"
}
```

Publishing the closed resource set is a design decision, not a convenience: an
agent that has to guess an identifier and be refused wastes a call and then
reasons about a refusal.

**An ungranted resource and a nonexistent one are indistinguishable** — same
code, same message, same status. Otherwise a caller could enumerate another
organisation's books by comparing responses.

> `test_an_unauthorised_portfolio_and_a_nonexistent_one_are_indistinguishable`

Two refusals the covenant handler owns, both because the evaluator cannot honour
the constraint rather than because policy forbids it:

* **an SPV-scoped resource** — the funded concentration path narrows by *book*,
  so answering an SPV grant would silently return the enclosing book;
* **a non-funded population** — concentration tests are defined on the funded
  book, so a `pipeline`-pinned resource is refused rather than answered from a
  population it does not name.

And one fail-closed check worth understanding. `mi_agent.portfolio_scope.apply_scope`
returns the frame **unchanged** when it carries no `source_portfolio_id` column.
For a UI lens that is a reasonable degradation. For a caller whose *authorisation*
is the narrowing it is a fail-open, so a book-scoped call first asks
`concentration_tests_api.funded_attribution_status` and refuses when the
attribution is absent.

*Known cost:* that check resolves the funded frame, so a book-scoped call
currently resolves it twice. It runs only when the resource actually narrows by
book, and it is a security control — correctness before performance for the
proof. Threading the resolved frames through is the obvious optimisation and is
not part of Sprint 1.

---

## 5. What an agent receives

The full `GovernedResult`, serialised:

```jsonc
{
  "capability": "tool.evaluate_covenants",
  "schema_version": "1.1.0",
  "status": "success",              // success | partial_success | blocked | error
  "request_id": "req_…",
  "correlation_id": "buyer-run-42", // echoes the caller's X-Correlation-Id
  "tenant_id": "ERE",
  "portfolio_id": "ERE/source_portfolio/direct_001",
  "snapshot": {                     // WHICH dataset answered
    "dataset_label": "…central_tape.csv",
    "snapshot_id": "snap_…", "content_hash": "sha256:…",
    "reporting_date": "2026-07-31", "row_count": 4821,
    "approval_state": "approved"
  },
  "result": {
    "tests": [ /* current_value, threshold, operator, utilisation, headroom,
                  breach_amount, status, prior_status, movement, provenance */ ],
    "summary": { "overall_status": "breach", "breaches": 1, … },
    "lineage": {                    // WHICH DEFINITION produced the numbers
      "configuration_version": 7, "configuration_hash": "cfg_…",
      "activated_by": "A. Operator", "library_version": "1.0.0"
    },
    "scope": { "source_portfolio_ids": ["direct_001"], … }
  },
  "policy": { "data_approved": true, "tenant_authorised": true, … },
  "audit":  { "organisation_id": "a2a_test_agent", "actor_type": "service", … },
  "error":  null
}
```

`status` and `error.code` are the contract; `message` is for humans. `blocked`
means governance refused — do not retry. `error` means the capability could not
produce an answer.

**The audit event carries none of it.** One structured line per execution on the
`trakt.audit` logger: capability, request id, correlation id, tenant,
organisation, actor, channel, resource, snapshot id, outcome, duration, error
code. Never the answer body, never a computed value, never loan rows.

---

## 6. Running it locally

```bash
source scripts/agent_dev_env.sh          # see config/dev/README.md
uvicorn mi_agent_api.app:app --port 8000

python scripts/agent_reference_client.py \
    --resource ERE/source_portfolio/direct_001 \
    --provider scripted            # or anthropic / openai
```

The reference client is an *outside* program: it has no Trakt imports at all and
reaches Trakt only over HTTP. `test_the_reference_client_imports_nothing_from_trakt`
parses its AST to assert that, because the claim is the whole point — an
in-process caller can always be given more than it should have.

`--provider scripted` runs the same workflow with **no model in the loop**. If it
completes, the permissions, the calculation, the evidence and the audit trail are
properties of Trakt rather than of whichever model happened to be driving. It is
also what runs in CI, where there is no API key.

---

## 7. Adding a tool

1. **Find the existing implementation.** If there isn't one, this is not a tool
   yet — the calculation belongs in the domain, with the UI and the agent both
   calling it.
2. Write a handler in `trakt_tools/handlers/`, importing heavy dependencies
   *inside* the function so the registry stays cheap to import.
3. Declare `INPUT_SCHEMA` / `OUTPUT_SCHEMA` with `trakt_tools.schema.object_schema`.
   Only the supported keyword subset is allowed — a keyword the executor cannot
   enforce is a registration failure, because a published constraint the server
   does not apply is worse than no constraint.
4. `register(ToolSpec(...))` in `trakt_tools/handlers/__init__.py`, naming a
   capability from `trakt_core.context.KNOWN_CAPABILITIES`. The grant vocabulary
   and the tool vocabulary are the same set; if you need a new verb, add it there
   first.
5. Regenerate the contract: `python scripts/build_agent_openapi.py`.
6. Add a test that the handler reaches the existing implementation and that its
   projection changes no value.

The registry is the single source for the REST surface, the OpenAPI document, a
future MCP server and the synthetic agents — so a tool cannot exist on one
surface and not another.

---

## 8. What is deliberately not here

| Not built | Why |
|---|---|
| MCP server | The registry is ready for one (`trakt_mcp/` would read the same declarations), but REST/OpenAPI is the minimum viable interface and has a settled auth story. Sprint 2 or later. |
| `get_loan` / `explain_value` | Sprint 2. They are the strategically important pair — value-level provenance is what turns a number into a fact — and they need the entity model first. |
| The other ~12 proposed tools | Sprint 2, and only where there is one authoritative implementation, it can be permissioned, its inputs/outputs can be typed, and its output can be audited. |
| `on_behalf_of` delegation | Needs an `ExecutionContext` field and an `AuditMetadata` field. Not required for a machine identity acting for an organisation. |
| Field-level authorisation | The two-axis model (scopes ∩ per-resource grants) covers everything through the synthetic demo. |
| Idempotency keys | No tool writes yet. Required before `raise_dd_request` and `record_decision` in Sprint 3. |
| An outbound notification seam | Trakt can be called but cannot call back. Named as a known gap; needed for event-driven A2A, not for this. |

---

## 9. Deployment requirements not enforceable from code

| # | Requirement | How to verify |
|---|---|---|
| A1 | Set `TRAKT_AGENT_API_ENABLED` only where the agent surface is intended. | `az webapp config appsettings list … --query "[?name=='TRAKT_AGENT_API_ENABLED']"` |
| A2 | Do **not** set `TRAKT_AGENT_AUTH_MODE=disabled` in any deployed environment. The guard refuses it in production, but the setting itself lives in Azure. | app settings must not contain it |
| A3 | Set `TRAKT_AGENT_ENTRA_AUDIENCE`, and register the agent application with the `Trakt.Agent` app role assigned to its service principal. | `GET /v1/agent/tools` with a role-less token must return 403 |
| A4 | Register the calling organisation and write its grants before enabling the surface. An agent with no grants holds no scopes and can do nothing — which is correct, but is better discovered in configuration review than by a client. | `GET /v1/agent/tools` returns a non-empty `tools` array |
