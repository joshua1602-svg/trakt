# A2A readiness — the design, and why none of it is built yet

*Sprint 2, Part 12. **Design only.** No `DDRequest`, no `DDResponse`, no Buyer
agent, no Seller agent, and no protocol implementation exists or should be
written from this document without a real counterparty.*

---

## What "A2A" would actually mean here

Not "two agents chat". The transaction Trakt sits inside is specific:

> A **buyer's** agent conducts diligence on a portfolio a **seller** holds in
> Trakt. It asks questions. It receives governed, evidenced answers, or governed
> refusals. Neither side's agent trusts the other's; both trust Trakt's envelope.

The interesting property is that this is **already almost the current
architecture**. A buyer's agent with an entitlement to a seller's resource,
calling `POST /v1/agent/tools/{name}`, receiving a `GovernedResult` with a
snapshot reference and provenance, is diligence over A2A in everything but name.

What is genuinely missing is not a protocol. It is four things underneath one.

---

## The four gaps, in the order they bite

### Gap 1 — cross-organisation entitlement has no lifecycle

The entitlement model is already the right shape: `organisation × resource ×
capability`, where the organisation need not be the resource's owner. A buyer
agent reading a seller's book is one grant.

What does not exist is how that grant comes to be, and how it ends:

- who proposes it, who approves it, and what record the approval leaves;
- **expiry** — a diligence grant is for a process with an end date, and a grant
  with no expiry is a permanent one that everybody forgot;
- revocation that takes effect on the next call rather than the next deploy;
- narrowing to a *purpose*, so the same grant cannot be reused for a different
  transaction next quarter.

This is the gap to close first, and it is closable without any protocol work at
all. It is configuration, an approval workflow and an expiry check.

### Gap 2 — the answers are per-question, and diligence is per-process

Every tool answers one question about one snapshot. A diligence exercise is a
process: hundreds of questions over weeks, against a book that publishes a new
snapshot each month, producing a record that has to be defensible after
completion.

The missing concept is an **enquiry** — a correlation-scoped, expiring container
that pins a snapshot, accumulates the calls made under it, and can be closed and
exported as the evidence pack.

Notably, most of this already exists in pieces: `correlation_id` threads through
every envelope and every audit event, `SnapshotRef` pins the dataset, and
`AuditMetadata` records each call. An enquiry is largely a *naming and a
lifecycle* over machinery that is already there — not new infrastructure.

### Gap 3 — a question Trakt cannot answer has no route to a human

Today a question outside the tool surface is a refusal. In a real diligence
process it is a **request**: the buyer asks something, the seller's team answers
or declines, and the exchange is part of the record.

This is the honest boundary of automation. Trakt should not invent an answer,
and it should not pretend the question was invalid. What it lacks is somewhere
for the question to go.

### Gap 4 — no counterparty exists to agree a protocol with

This is the reason nothing is built. A protocol is an agreement, and there is no
second party. Every wire format designed in this condition is a guess that will
be replaced by whatever the first real counterparty already uses — and by then it
will have accreted callers.

---

## The intended topology

    Buyer / Funder Agent
    │
    │  A2A            (agent ↔ agent: discovery, identity, task state)
    ▼
    Seller / Originator Agent
    │
    │  MCP            (agent ↔ tools: declaration and invocation)
    ▼
    Trakt

Two protocols doing two different jobs, and the distinction is the point. **MCP
is how an agent reaches tools.** **A2A is how two agents reach each other.**
Trakt sits at the bottom and should speak the tool protocol; it should never
become a party to the negotiation above it.

Trakt's position in that diagram is also why it needs almost nothing new: the
seller's agent is just a client agent (`docs/a2a_agent_boundary.md`), and a
buyer's agent granted an entitlement is another one.

---

## The nine questions, answered against what exists

| Concern | Status today | What A2A would add |
|---|---|---|
| **Agent discovery** | ❌ none | An agent card at a well-known URL: who this deployment is, and how to request access. Trakt currently assumes the caller already knows the endpoint. |
| **Agent identity** | ✅ **solved** | Entra service principal → `ExecutionContext(actor_id, actor_type=service, organisation_id, channel=enterprise_agent)`. Machine identity is first-class, not a human account lent to a bot. |
| **Capability advertisement** | ✅ **solved** | `GET /v1/agent/tools` already returns the caller-narrowed tool list *and* the closed set of resource identifiers it may name. This is exactly what a capability advertisement is. |
| **Task / request structure** | ◐ per-call only | JSON Schema in, `GovernedResult` out — complete for one question. Missing is the *enquiry* that groups many (Gap 2). |
| **Authentication** | ✅ **solved** | OIDC/JWKS bearer, issuer and audience validated, `Trakt.Agent` app role required, disabled mode refused in production. |
| **Organisation ownership** | ✅ **solved** | `organisation × resource × capability`. The organisation need not own the resource — the property that makes cross-institution access a grant rather than a rewrite. |
| **Correlation IDs** | ✅ **solved** | `X-Correlation-Id` in, threaded through `GovernedResult.correlation_id` and `AuditMetadata.correlation_id`. Both sides can join their records of the same exchange. |
| **Long-running task state** | ❌ none | Every call is synchronous. A diligence sweep over a large book, or a referral awaiting a human, needs a task that can be polled. |
| **Evidence references** | ✅ **solved** | `SnapshotRef` (id + content hash), `ProvenanceRef`, and `explain_values` down to the valuation observation and the policy version that selected it. |

**Six of nine already hold.** The three that do not — discovery, enquiry
grouping, long-running state — are all *additive*: none requires changing a tool
contract, and none requires a counterparty to build.

---

## The design, at the level it is safe to hold now

### The transport is HTTP with typed envelopes, and it already exists

The `GovernedResult` envelope is the A2A message. It carries what a
counterparty's agent actually needs to trust an answer:

| Field | Why a counterparty needs it |
|---|---|
| `capability` | what was asked |
| `status` | `success` / `blocked` / `error` — a refusal is not an empty answer |
| `snapshot` | which dataset, with its content hash |
| `policy` | under which approval state |
| `provenance` | what backs the figures |
| `correlation_id` | which enquiry this belongs to |
| `audit` | the seller's own record of the same call |
| `error.retryable` | whether to stop — a governance decision must not be retried |

Nothing in that list is A2A-specific. It is what any consumer needs, which is
precisely why it should not be re-specified for A2A.

### The message pattern, if and when a counterparty exists

    buyer agent  --(1) enquiry.open ----------------> Trakt (seller's deployment)
                 <-(2) enquiry ref + pinned snapshot--
                 --(3) tool call, correlation=enquiry->
                 <-(4) GovernedResult ----------------
                       ... repeated ...
                 --(5) enquiry.close ---------------->
                 <-(6) evidence pack -----------------

Steps 3 and 4 exist today, complete. Steps 1, 2, 5 and 6 are Gap 2. Anything the
buyer asks that Trakt cannot answer becomes a referral (Gap 3), which appears in
the pack as an open item rather than as a silence.

### What must NOT be built

- **A proprietary Trakt A2A protocol.** If a standard emerges — MCP has server-to-
  server ambitions, and other agent-interop work is live — Trakt should adopt it.
  `trakt_tools/mcp.py` is deliberately shaped to show how little that costs when
  the tool surface is a registry rather than a set of endpoints.
- **A negotiation layer.** Price, terms and structure are the parties'; Trakt
  supplies evidence.
- **An LLM anywhere in the answer path.** The buyer's agent may be a model and
  the seller's may be a model. What passes between them must be deterministic,
  reproducible and attributable, or the whole exercise is two models agreeing
  with each other.

---

## Ordering, and the honest cost

| | Work | Depends on a counterparty? |
|---|---|---|
| 1 | Entitlement lifecycle: proposal, approval, expiry, revocation, purpose | No |
| 2 | The enquiry: open / correlate / close / export | No |
| 3 | Referral: a route for a question Trakt cannot answer | No |
| 4 | Protocol adoption | **Yes** |

Three of the four are buildable now and are valuable on their own — an expiring,
approved, revocable grant and an exportable evidence pack are worth having
whether or not a counterparty ever appears. The fourth should wait, and this
document exists so that waiting is a decision rather than an omission.

---

## What Sprint 2 must preserve for A2A to be addable later

Five contract properties. Each is true today; each would be expensive to
reintroduce once tools have external callers.

1. **Every tool names a `resource`, and it is required.** Enforced at
   registration (`ToolSpec.__post_init__` rejects a spec whose resource argument
   is optional). A tool that could default its resource could not be granted
   across organisations, because there would be nothing to grant.

2. **Identity never comes from arguments.** Every input schema is closed, and
   `trakt_tools.mcp.refuse_identity_in_arguments` makes the rule enforceable at
   an adapter boundary rather than merely documented.

3. **`correlation_id` is carried, never generated when supplied.** It is what
   turns a sequence of calls into an enquiry later, and a Trakt that overwrote a
   caller's correlation id would make the two sides' records unjoinable.

4. **Refusals are typed and carry `retryable`.** An autonomous counterparty must
   be able to tell "I am not allowed to ask that" from "that could not be
   computed" without parsing prose — and must stop rather than loop on the first.

5. **Every answer carries its snapshot.** A diligence answer without the dataset
   it came from cannot be re-verified after the fact, which is the whole point of
   the exercise.

Adding a version-2 tool later is fine; a version-1 tool that silently changes
what it means is not. `ToolSpec.version` exists from the first tool onward for
exactly this reason.

---

## The one thing to keep true in the meantime

Every capability added between now and then should be **usable by an
organisation that does not own the data**. That is the only property that makes
A2A a configuration change rather than a rewrite, and it is a property that is
cheap to preserve and expensive to retrofit.

Concretely, it means a tool must never assume the caller is the tenant, and a
capability must never be implied by ownership. Both are already true — this is
about keeping them true.
