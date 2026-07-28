# 03 — Workflow diagrams

Three operational workflows. All three run through the same Workflow State
Engine and the same stage graph; they differ in which rules are pre-applied and
which decisions surface.

## 0. Workflow selection (automatic)

```mermaid
flowchart TD
    A[Delivery received /\noperator starts a workflow] --> B{Client known?\nsource registry +\nclient memory}
    B -- no --> C[Workflow 1\nInitial client onboarding]
    B -- yes --> D{Schema fingerprint matches\nan active SourceRecord?}
    D -- no --> E[Workflow 2\nNew portfolio onboarding]
    D -- yes --> F[Workflow 3\nRecurring reporting]
```

This reuses the classification the blob trigger already performs
(`new_source` / `schema_drift` / deterministic route) — surfaced to the
operator as "new client" / "new portfolio" / "regular delivery".

## 1. Initial client onboarding

```mermaid
flowchart TD
    A[Operator: Start something new\nchooses outcome + uploads tapes] --> B[Received\nfiles registered, workflow run created]
    B --> C[Understanding data\nonboarding agent profiles files]
    C --> D{Mapping\nautomatic matches +\nsuggestions}
    D -- decisions needed --> E[Review Centre\nconfirm mappings, aliases, enums\nchoose scope per decision]
    E --> F[Rules persisted\nrule store + projector]
    F --> G[Rerun affected stage\norchestrator resume]
    G --> D
    D -- all resolved --> H[Validation\nvalidation agent]
    H -- warnings/exceptions --> E
    H -- clean --> I[Projection]
    I --> J[Assembly]
    J --> K[Publication review\nreconciliation + summary]
    K -- operator approves --> L[Published\npromoted to production latest]
    K -- operator rejects --> M[Workflow parked\nwith reason]
```

Operator experience: guided, stage by stage; every stop is a Review Centre
decision, never a technical screen.

## 2. New portfolio onboarding (existing client)

```mermaid
flowchart TD
    A[New portfolio delivery\nfor an approved client] --> B[Received]
    B --> C[Apply approved client rules\nmappings, aliases, enums,\nclient memory - automatic]
    C --> D[Understanding data\nonly unmatched / new items remain]
    D --> E{Portfolio-specific\ndecisions?}
    E -- yes --> F[Review Centre\nnew schema fields, new products,\nnew enum values only]
    F --> G[Rules persisted\nscope: portfolio or client,\noperator chooses]
    G --> H[Rerun affected stage]
    H --> E
    E -- none left --> I[Validation → Projection → Assembly]
    I --> J[Publication review]
    J -- approve --> K[Published + SourceRecord promoted\nfuture deliveries route automatically]
```

Key property: approved client-scoped rules are **pre-applied and never
re-reviewed**; only deltas surface. The UI clearly labels the run
"Existing client — new portfolio".

## 3. Recurring reporting

```mermaid
flowchart TD
    A[Monthly delivery arrives] --> B[Received\nrecognised automatically]
    B --> C[All approved rules applied\nfull pipeline runs unattended]
    C --> D{Anything new?\nnew fields, new values,\nnew warnings, material\nvalidation changes}
    D -- no --> E[Report prepared\nawaiting publication approval]
    D -- yes --> F[Review Centre\nexceptions only]
    F --> G[Decisions persisted as rules\nrerun affected stage]
    G --> D
    E --> H{Operator approves\npublication?}
    H -- yes --> I[Published to production latest\nhistory + rule versions recorded]
    H -- no --> J[Held with reason]
```

Onboarding is never repeated. A clean month is two clicks: open, approve.

## 4. Historical artefacts (backfill)

Historical deliveries after onboarding follow Workflow 3's shape: approved
rules auto-apply per delivery; the operator reviews only differences between
the historical file and the approved rule set. Runs can be queued per period
and reviewed from the same Review Centre.

## 5. Decision loop (common to all workflows)

```mermaid
flowchart LR
    A[LLM or deterministic\nsuggestion] --> B[Deterministic validation\nbackstop statuses]
    B --> C[Operator review\nReview Centre]
    C -- approve + scope --> D[Persist rule\nversioned, scoped, audited]
    D --> E[Project rule into\nagent-readable sinks]
    E --> F[Rerun affected stage]
    C -- reject --> G[Record rejection\naudit only, no rule]
```

No suggestion ever updates a registry without passing through this loop.
