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

## The one thing to keep true in the meantime

Every capability added between now and then should be **usable by an
organisation that does not own the data**. That is the only property that makes
A2A a configuration change rather than a rewrite, and it is a property that is
cheap to preserve and expensive to retrofit.

Concretely, it means a tool must never assume the caller is the tenant, and a
capability must never be implied by ownership. Both are already true — this is
about keeping them true.
