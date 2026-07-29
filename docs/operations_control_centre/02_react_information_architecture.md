# 02 — React information architecture

New app: `frontend/operations-control-ui/`. Stack identical to
`frontend/mi-agent-ui` (Vite 6, React 18, TypeScript, Tailwind v4, Vitest,
`AgentClient` Http/Mock/Caching pattern), plus `react-router` for shareable
deep links to a workflow or a decision.

Target user: an intelligent business user with no technical knowledge.
Guiding rule: **every screen answers one question — "what do I need to decide
next?"**

## 1. Navigation map

```
/                        Dashboard            (landing page)
/workflows               All workflows        (list, filterable)
/workflows/:id           Workflow screen      (stage tracker + stage detail)
/reviews                 Review Centre        (every outstanding decision)
/reviews/:id             Decision detail      (question, recommendation, evidence, scope)
/rules                   Rules Library        (search approved rules)
/rules/:id               Rule detail          (history, approvals, scope)
/history                 Reporting history    (published versions, comparisons, rollback)
/history/:client/:period Delivery detail
/new                     Start something      (outcome picker — the only entry to new work)
```

Primary navigation: five items — **Dashboard · Review · Workflows · Rules ·
History** — plus one action button: **"Start something new"**. Nothing else.

## 2. Screens

### 2.1 Dashboard (`/`)

A calm operational summary. Six tiles maximum, each a count + one line, each
clicking through to a filtered list:

- New deliveries received
- Onboardings awaiting review
- Monthly reports awaiting approval
- Blocked workflows
- Recently published portfolios
- Recent approvals

No charts, no technical status, no logs.

### 2.2 Start something (`/new`)

The operator chooses **outcomes**, never agents:

> **What would you like Trakt to prepare?**
> ○ MI Reporting
> ○ MI Reporting + ESMA Annex 2

Then: choose client (or "This is a new client"), upload/point to the delivery
(loan tape, collateral tape, cash flow tape, limits, supporting artefacts —
each slot labelled in business language with a short "what is this?" hint).
The Control Centre determines the workflow type (initial onboarding /
new portfolio / recurring) and says so in one sentence before starting:

> "We know this client. This looks like a new portfolio, so Trakt will reuse
> the rules you've already approved and only ask about what's new."

### 2.3 Workflow screen (`/workflows/:id`)

Header: client, portfolio, outcome, one-line status sentence.

Stage tracker (horizontal steps, operator vocabulary only):

```
Received → Understanding data → Mapping → Validation → Projection → Assembly → Publication
```

Each stage chip shows one of: **Complete · Needs Review · Blocked · Waiting ·
Approved · Rejected** (rendered from the Governed Agent Result status).

Selecting a stage shows its Governed Agent Result:
1. **Summary** (1–2 sentences)
2. **Why this matters** (business explanation)
3. **Decisions required** (cards; the only interactive part)
4. **Evidence** (collapsed accordion, expandable)

Footer actions, shown only when applicable: "Review decisions" (goes to the
Review Centre filtered to this workflow), "Run again", "Approve publication".

### 2.4 Review Centre (`/reviews`)

One queue of every outstanding decision across all workflows. Each row:

- Plain-language title ("Confirm where 'Prop_Val_Idx' belongs")
- Client / portfolio
- What kind of decision (Confirm mapping · Review validation warning ·
  Approve reconciliation · Approve publication)
- Whether it is blocking

Only new / changed / material / exception items ever appear. Items approved
previously and unchanged never re-enter the queue.

### 2.5 Decision detail (`/reviews/:id`)

The heart of the product. One decision per screen:

1. The **question**, in one sentence.
2. Trakt's **recommendation**, with provenance in words:
   "Suggested automatically (high confidence)" / "Suggested by Trakt's
   assistant — please check" / "Matches a rule you approved in March".
3. **Options** (radio/select), or Approve / Reject / Edit.
4. **Scope picker** — "Apply this decision to:"
   ○ This file only · ○ This portfolio · ○ This client · ○ All of Trakt
   with one line explaining the default and consequences.
5. **Evidence**, collapsed: sample values, before/after preview, affected count.
6. On submit: confirmation sentence + "Trakt is re-running the affected step."

### 2.6 Rules Library (`/rules`)

Search + filters (kind: mappings / aliases / enums / transformations /
portfolio rules / client rules / exceptions / reporting assumptions; scope;
client). Each rule row: plain description, scope badge, approved-by, date.
Rule detail: current version, full version history, the approvals that created
each version, where it has been applied.

### 2.7 Reporting history (`/history`)

Per client/portfolio: reporting dates, published versions, the rule versions
each publication used, publication date and approver, comparison between two
deliveries ("what changed since last month" in business terms), and rollback
availability ("Republish the previous version") — rollback itself is an
approval-gated action.

## 3. UX language rules

- Statuses, stage names and decision kinds come from a single shared copy
  module; no component invents wording.
- Never rendered: JSON, stack traces, paths, container names, schema/regime
  codes, agent names, Python errors. API failures render as
  "Something went wrong on our side. Nothing has been lost — try again in a
  moment." with a reference code for support.
- Spacious layout: one primary action per screen, generous whitespace, no
  dense tables outside evidence accordions.
- Empty states are reassuring, not blank ("Nothing needs your attention.").

## 4. Frontend technical notes

- `src/api/OpsClient.ts` interface + `HttpOpsClient` / `MockOpsClient` /
  caching wrapper, mirroring `mi-agent-ui`'s `AgentClient` triad; base URL via
  `VITE_OPS_API_URL`, mode via `VITE_OPS_MODE`.
- Run status by polling (`GET /ops/workflows/:id`, interval with backoff) —
  consistent with the codebase's synchronous/poll model; no websockets.
- State: one `useOpsWorkspace` hook + router state; localStorage persistence
  for non-sensitive UI preferences only.
- All copy testable: contract tests assert forbidden-vocabulary rules
  (doc 07 §4) against rendered strings.
