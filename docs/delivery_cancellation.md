# Cancelling a delivery

A delivery workflow could be approved or held, never ended. Everything needed
to end one existed — `OpsEngine.cancel`, `POST /ops/workflows/{id}/cancel`, the
`cancelled` status and its transitions from every non-terminal state — and
nothing in the OCC called any of it. A delivery raised by mistake stayed on the
list, and its open questions stayed in the review queue, for good.

## What cancelling now does

```
Cancel this delivery
        ↓
Confirm, with a reason      ← required, and kept
        ↓
Status → cancelled          ← a governed transition, refused after publication
Open questions → superseded ← it stops asking
Off the working list        ← kept, findable under "Cancelled"
Audit entry per change      ← the hash chain still verifies
```

**Nothing is deleted.** The delivery, its steps, its files and its history are
all still readable. It simply stops needing anything from anybody.

## The three changes

### 1. It stops asking

`OpsEngine.cancel` now closes the decisions the run still had open. They are
marked **superseded**, not approved or rejected: neither answer was given, and
recording one would put a decision in the audit trail that nobody made. The
status existed in `contracts.py` and had never been used — this is what it is
for.

Without this, cancelling a delivery would leave its questions in the review
queue for a workflow nobody can act on, which is the opposite of the point.

### 2. It leaves the working list

`GET /ops/workflows` with no status filter no longer returns cancelled runs.
"All" means everything still live. A **Cancelled** filter finds them, and
`?status=cancelled` returns them over the API, so nothing has disappeared.

### 3. There is a button

On the delivery screen, below everything else and low emphasis — cancelling is
always available on a live delivery and never the suggested move. It confirms
first, requires a reason, and states what is *not* happening: nothing has been
published, so nothing is withdrawn.

It is not offered on a published or already-cancelled delivery. The engine
refuses those transitions anyway; the UI simply agrees with it.

## Clearing a backlog

Cancelling one at a time is right for real work and wrong for clearing forty
test runs. For that:

```bash
python scripts/cancel_test_workflows.py --client ERE                    # dry run
python scripts/cancel_test_workflows.py --client ERE --status needs_review
python scripts/cancel_test_workflows.py --client ERE --before 2026-07-01 --apply
```

It goes through `OpsEngine.cancel`, so every cancellation is the same governed,
audited transition the button produces — not a delete. It is a dry run until
`--apply`, and it never touches a published delivery.

## Why not published

Cancelling a published delivery would imply the report that went out came back.
It did not. Withdrawing something already published is a different act with a
different record, and pretending cancellation covers it would put a false
statement in the audit trail.

## What proves it

| Level | Tests |
|---|---|
| Engine | Cancelling closes the questions it was still asking; a closed question is superseded, not answered; the run, its events and the hash-chained audit all survive; a published delivery cannot be cancelled |
| API | A cancelled delivery leaves the working list but is still returned by `?status=cancelled`; the delivery itself is still readable with its history intact; it stops appearing in `/ops/reviews`; publication blocks cancellation; the reason is on the audit record |
| Browser | The action is offered on a live delivery and asks why; the reason is required; cancelling ends it and clears its questions; it leaves the list and stays findable; it is not offered once published |

Screenshots: [`docs/screenshots/delivery_cancellation/`](screenshots/delivery_cancellation/).
