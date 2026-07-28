# 04 — State diagrams

Five state machines, all persisted in `operations-control/` (doc 06). Every
transition writes an audit event; illegal transitions are rejected by the
Workflow State Engine, not left to the UI.

## 1. Workflow run

```mermaid
stateDiagram-v2
    [*] --> received : delivery registered / operator starts
    received --> running : engine invokes orchestrator
    running --> needs_review : stage returns open decisions
    running --> blocked : stage blocking / failure
    needs_review --> running : decisions resolved → rerun affected stage
    blocked --> running : blocker resolved → rerun
    blocked --> cancelled : operator abandons (with reason)
    running --> awaiting_publication : all stages complete
    awaiting_publication --> published : publication approved
    awaiting_publication --> held : publication rejected (with reason)
    held --> awaiting_publication : re-submitted after changes
    published --> [*]
    cancelled --> [*]
```

Recovery: `running` is resumable at any time from persisted state
(orchestrator `run_state.json` + workflow record); a crashed run re-enters
`running` via rerun without repeating completed stages.

## 2. Stage (within a workflow run)

Operator-facing statuses (doc 07 §2):

```mermaid
stateDiagram-v2
    [*] --> waiting
    waiting --> running : upstream stage complete
    running --> completed : readiness flag true, no open decisions
    running --> needs_review : open decisions, non-blocking
    running --> blocked : blocking issues / handoff refusal
    needs_review --> approved : all decisions approved
    needs_review --> rejected : operator rejects proposal
    approved --> running : affected stage rerun
    blocked --> running : rerun after resolution
    rejected --> [*]
    completed --> [*]
```

`waiting` renders as **Waiting**, `running` as a quiet in-progress indicator;
the operator vocabulary is exactly: Complete · Needs Review · Blocked ·
Waiting · Approved · Rejected.

## 3. Review item (decision)

```mermaid
stateDiagram-v2
    [*] --> open : emitted by a Governed Agent Result
    open --> approved : operator approves (records scope)
    open --> rejected : operator rejects (reason required)
    open --> superseded : stage rerun changed the underlying item
    approved --> [*] : rule persisted + stage rerun scheduled
    rejected --> [*] : audit only
    superseded --> [*] : replaced by a new open item if still relevant
```

An item already approved never re-opens unless the underlying data changed —
in that case a **new** item is raised marked "changed since your approval".

## 4. Rule (persistent operational decision)

```mermaid
stateDiagram-v2
    [*] --> proposed : from an approved review item
    proposed --> active : persisted + projected into agent sinks
    active --> superseded : new version approved (scope/value change)
    active --> retired : operator retires (reason required)
    superseded --> [*]
    retired --> [*]
```

Versioning: `superseded` versions are immutable and permanently queryable;
each publication records the exact rule versions it used (doc 06/08).

## 5. Publication

```mermaid
stateDiagram-v2
    [*] --> prepared : assembly + delivery checks complete
    prepared --> approved : operator approves publication
    prepared --> rejected : operator rejects (reason required)
    approved --> published : promote to production latest\n(existing promote path)
    published --> rolled_back : operator approves rollback\n(re-publish prior version)
    rejected --> prepared : workflow amended and re-prepared
    rolled_back --> [*]
    published --> [*]
```

Invariant: no transition into `published` exists except from an explicit
operator `approved`. Agent completion can only ever reach `prepared`.
