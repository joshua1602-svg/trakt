# The reference agent and the client agent

*Sprint 2, Part 11. Written before a client agent exists, because the boundary is
cheap to hold now and expensive to recover later.*

---

## The claim this document makes falsifiable

> Trakt's reference agent has **no privileged access**. Anything it can do, a
> client-owned agent can do, through the same door, under the same governance.

That is easy to say and easy to lose. It is lost the first time someone adds a
convenience: an internal import "just for the reference agent", a tenant header
the internal caller may set, a tool that is not in the published catalogue. Each
is individually reasonable. Together they mean the reference agent is a
demonstration of something Trakt cannot actually sell.

So the boundary is stated here as a small number of properties, each with the
test that holds it.

---

## The two agents

| | Reference agent | Client agent |
|---|---|---|
| **Who owns it** | Trakt | The client |
| **Who runs it** | Trakt's infrastructure | The client's infrastructure |
| **What it is** | An example, and the Securitisation Readiness Agent | An enterprise agent doing the client's own work |
| **Model** | Whichever the operator configures | Whichever the client chooses — Trakt never knows |
| **How it reaches Trakt** | `POST /v1/agent/tools/{name}` over HTTPS | The same |
| **How it authenticates** | Entra service principal, `Trakt.Agent` role | The same |
| **What it may name** | Resources granted to its organisation | The same |
| **What it may call** | Tools in `GET /v1/agent/tools` | The same |
| **What it receives** | `GovernedResult` JSON | The same |
| **Audit** | One event per call, by correlation id | The same |

The right-hand column is "the same" nine times. That is the whole point: the only
genuine differences are ownership and operation.

---

## The five properties, and what holds each

### 1. The reference agent imports nothing from Trakt

It is a separate program that speaks HTTP. Not "should be" — asserted.

`tests/test_agent_reference_client.py::test_the_reference_client_imports_nothing_from_trakt`
parses `scripts/agent_reference_client.py` with the `ast` module and fails on any
import of `trakt_core`, `trakt_tools`, `mi_agent`, `mi_agent_api`, `engine` or
`analytics_lib`. A companion test asserts it makes no call to the builtins
`open`, `exec`, `eval` or `__import__` — because "no import statement" is not the
same claim as "no access to the filesystem".

If that file could reach inside Trakt, the demonstration would prove nothing
about an external agent.

### 2. There is one door, and it is the published one

Every agent call goes through `execute_governed_tool`, in a fixed order: tool
exists → schema → capability → entitlement → source approval → handler → envelope
→ audit. There is no internal bypass and no second entry point.

`tests/test_agent_governed_execution.py` asserts the ordering directly, including
that **no dataframe is resolved** for a caller who fails any check before step 6.
The refusal costs a schema validation, not a data read.

### 3. Identity comes from the transport, never from the payload

`ExecutionContext` is built from a validated Entra token — tenant, organisation,
actor, channel, scopes. A tool argument named `tenant_id` is not read; it is
refused (`trakt_tools.mcp.refuse_identity_in_arguments`, and the HTTP surface has
no such argument to begin with because every input schema is closed).

This is why "the same door" is a security statement and not just an architectural
one. A reference agent that could assert its own tenant would be a privilege the
client agent does not have — and a vulnerability the client agent could exploit.

### 4. The catalogue is a capability statement, not a menu

`GET /v1/agent/tools` returns the tools this caller may actually use, narrowed by
capability, plus the closed set of resource identifiers it may name. Both agents
read the same endpoint and get an answer narrowed to themselves.

An agent offered a tool it will always be refused wastes a call and then reasons
about a refusal. Worse, an agent that must *guess* a resource identifier will
guess, and a guessed identifier that happens to exist is a probe.

### 5. Trakt is model-agnostic, on both sides

Trakt publishes JSON Schema. Translation into a provider's tool format happens in
the *client*, at its own boundary — `scripts/agent_reference_client.py` has one
adapter class per provider and nothing provider-specific reaches Trakt.

The `scripted` provider exists to prove the point the model-driven ones cannot:
the same workflow completes with **no model in the loop at all**. If it does, then
the permissions, the calculations, the evidence and the audit trail are properties
of Trakt rather than of whichever model happened to be driving.

---

## What the reference agent may legitimately have

Not everything is symmetric, and pretending otherwise would be its own dishonesty.

- **A different organisation, with different grants.** Trakt's own organisation
  may be entitled to more resources than a given client's. That is the
  entitlement model working, not a bypass: the *mechanism* is identical and the
  grants are data.
- **Operational access Trakt has as the operator** — logs, deployment,
  configuration review. None of it flows through the agent API, and none of it is
  reachable from the agent's code.
- **Earlier sight of new tools.** A tool can be registered and not yet granted to
  any client organisation. It is still in the same registry, still governed
  identically, and still invisible to a caller without the capability.

The line: **a difference in what an organisation is granted is legitimate; a
difference in how the request is governed is not.**

---

## What would break the boundary

Recorded so it is recognisable in review:

1. A tool handler that behaves differently based on `context.organisation_id`.
2. An internal function the reference agent calls that skips
   `execute_governed_tool`.
3. Any argument that can influence identity, scope or tenancy.
4. A tool registered outside `trakt_tools.registry` — it would be absent from the
   catalogue, the OpenAPI document and the MCP declarations, and present only to
   whoever knew its name.
5. A "development mode" that relaxes entitlement rather than relaxing
   *authentication*. `TRAKT_AGENT_AUTH_MODE=disabled` exists and is refused in
   production (`mi_agent_api/agent_auth.py`); it weakens who you must prove you
   are, never what that identity may then reach.

---

## Consequence for the Securitisation Readiness Agent

It is a **client agent that Trakt happens to own**. It gets no special tool, no
special access and no special path — which is what makes it a credible
demonstration to a client evaluating whether to build their own.

If the Securitisation Readiness Agent needs a capability Trakt does not yet
expose, the answer is to add a governed tool that every entitled organisation can
call. It is never to give the reference agent a shortcut.
