# 01 — Architecture proposal

## 1. Objective

Introduce a governed operational layer — the **Operations Control Centre** —
above the existing Trakt agents. The layer owns workflow orchestration, human
approvals, persistent decisions, operational visibility, publication, audit,
recovery and reruns. The existing agents keep all business logic and are not
modified.

```
React Operations Control Centre        (new — frontend/operations-control-ui)
        ↓ HTTPS
Operations Control API                 (new — operations_control/api, FastAPI)
        ↓ in-process
Workflow State Engine                  (new — operations_control/engine)
        ↓ adapters
Existing Agents                        (unchanged — engine/*, apps/blob_trigger_app)
        ↓ manifests + artefacts
Governed Agent Results                 (new adapter output — operations_control/contracts)
        ↓
Operations Control Centre              (renders status, decisions, evidence)
```

The React application never invokes Python scripts. It only talks to the
Operations Control API.

## 2. What already exists, and what we reuse

The codebase already contains most of the raw material for this layer. The
Control Centre **composes** it rather than rebuilding it.

| Capability | Existing asset | How the Control Centre uses it |
|---|---|---|
| Agent conductor | `engine/orchestrator_agent/` — `run_orchestration()`, resumable `RunState` (`run_state.json`), exit codes 0/2/3 | Invoked by the Workflow State Engine as the single pipeline entry point. Untouched. |
| Agent seam | `engine/orchestrator_agent/adapters.py` — `AgentAdapters` with `onboard/transform/validate/stamp/assemble/route_mi/project`, each returning `StepResult(ok, blocking, readiness, blockers, manifest_path, message)` | Wrapped (subclassed/decorated) by new **governed adapters** that translate every `StepResult` + numbered manifest into a Governed Agent Result. No agent code changes. |
| Stage status | Numbered manifests: onboarding `24_*`, transformation `30_*`, validation `40_*`, projection `50_*`, delivery `60_*`, plus the readiness ladder (`ready_for_*` flags, `next_agent`) | Read-only manifest readers per stage. The readiness ladder is the source of truth for stage status. |
| Onboarding decisions | `34_target_first_decisions.yaml`, `33_mapping_review_queue.*`, `40_operator_workflow_summary.json` (statuses `READY / NEEDS_CONFIRMATION / NEEDS_CONFIGURATION / BLOCKED / FAILED`) | Surfaced as Review Centre items; approvals written back through the same artefacts the agents already re-read on rerun. |
| Source approvals & promotion | `apps/blob_trigger_app/approvals.py` (`pending → approved/rejected → promoted`), `run_records.py`, `ops.py rerun` | Reused as-is for new-source / schema-drift approvals and for the publication promote step. |
| Governed envelope | `trakt_core/envelope.py` — `GovernedResult`, `AuditMetadata`, `TraktError`; `trakt_core/{context,policy,tenancy,runtime}.py` | The Operations Control API responds in `GovernedResult` envelopes; the new Governed **Agent** Result rides inside as the `result` payload. |
| API scaffold | `mi_agent_api/` — `auth_guard` (Easy Auth / SWA principal, `operator` role), `gateway.py` prefix handling, `TraktError` exception handlers, CORS, lifespan | Copied as the scaffold for the new `operations_control/api` FastAPI app. |
| React stack | `frontend/mi-agent-ui/` — Vite 6, React 18, TypeScript, Tailwind v4, `AgentClient` interface with Http/Mock/Caching implementations | Same stack and API-client pattern for `frontend/operations-control-ui`. |
| Client rule memory | `engine/onboarding_agent/mapping_memory.py` (client-scoped mapping/enum/precedence memory), `mapping_persistence.py` (controlled sinks: client memory / `aliases_pipeline.yaml` / `fields_registry_pipeline.yaml`) | The **rule projector** persists approved rules by writing to these existing, already-governed sinks. The core registry is never touched. |
| Storage abstraction | `apps/blob_trigger_app/storage.py` (`blob://` URIs, `TRAKT_STORAGE_BACKEND` file/blob duality) | Reused for the new `operations-control` container so local dev works file-backed. |
| Immutable audit ledger | `exception_db.py` hash-chained remediations, `export_audit_pack.py` | Pattern reused: Control Centre audit events are hash-chained JSON in the new container. |

## 3. New components

All new code lives in two new top-level trees. Nothing inside `engine/`,
`apps/`, `mi_agent*`, `trakt_core/` or `config/` is modified.

### 3.1 `operations_control/` (new Python package)

```
operations_control/
    contracts/      GovernedAgentResult, DecisionRequired, Evidence, RuleRecord (doc 07, 08)
    engine/         Workflow State Engine: workflow definitions, stage graph,
                    run lifecycle, rerun/recovery, publication gating (docs 03, 04)
    adapters/       Governed adapters around AgentAdapters + manifest readers
                    (one per agent: onboarding, transform, validation,
                    projection, assembly, delivery); plain-language translators
    stores/         operations-control blob container access: workflow runs,
                    governed results, approvals, rules, audit, history (doc 06)
    rules/          Rule store, scoping, versioning, and the rule projector that
                    writes approved rules into existing agent-readable sinks (doc 08)
    api/            FastAPI app (doc 05) — routers: dashboard, workflows,
                    reviews, rules, history, publications; auth reused from the
                    mi_agent_api pattern
```

### 3.2 `frontend/operations-control-ui/` (new React app)

Same stack as `mi-agent-ui` (Vite 6 + React 18 + TS + Tailwind v4 + Vitest),
plus `react-router` for deep links to a workflow or review item (the one
deliberate addition — operators will share links to "this approval").
Screens in doc 02.

## 4. How a workflow run works (non-invasive execution)

1. Operator chooses an **outcome** ("MI Reporting", "MI Reporting + ESMA
   Annex 2") — never an agent. The Workflow State Engine maps the outcome to an
   orchestrator target (`mi`, `regime`, `all`) and workflow type (initial
   onboarding / new portfolio / recurring reporting).
2. The engine creates a **workflow run** record in `operations-control/` and
   invokes `run_orchestration()` with governed adapters injected, writing agent
   outputs to a run-scoped staging directory — never directly to
   `processed-v2/.../latest`.
3. Each `StepResult` is translated into a **Governed Agent Result** (doc 07)
   and persisted. `blocking=True` → the workflow parks in *Needs Review* and
   the outstanding decisions appear in the Review Centre.
4. Operator decisions are written to the artefacts agents already understand
   (`34_*_approved.yaml`, approvals JSON, client memory via the rule
   projector), each simultaneously recorded as a scoped, versioned **rule**
   (doc 08) plus an audit event.
5. The engine reruns the affected stage using the orchestrator's existing
   `--resume` capability. Only stages downstream of the decision re-execute.
6. When all stages complete, the workflow parks at **Awaiting publication**.
   Publication is an explicit approval that triggers the existing promote path
   (`approvals.promote` / assembler outputs copied to `processed-v2/.../latest`).
   Nothing is published automatically.

## 5. Workflow selection logic (client vs portfolio vs recurring)

The engine distinguishes the three workflows using data that already exists:

- **Completely new client** — no client entry in the source registry
  (`trakt-state/registry/source_registry.yaml`) and no client memory directory.
  → Initial onboarding workflow: full guided onboarding, all decisions open.
- **Existing client, new portfolio** — client known, but the delivery's schema
  fingerprint / portfolio id has no active `SourceRecord`. → Secondary
  onboarding: client-scoped rules (mapping memory, approved aliases, approved
  enums) are pre-applied; only portfolio-specific deltas surface for review.
- **Recurring delivery** — active `SourceRecord` with matching schema
  fingerprint. → Recurring reporting workflow: fully automatic run; only new
  fields, new values, new warnings and material validation changes surface.

This mirrors the classification the blob trigger already performs
(`new_source` vs `schema_drift` vs deterministic route) — the Control Centre
lifts it into an operator-visible concept instead of inventing a parallel one.

## 6. LLM governance

Unchanged from today's advisory-only posture, made visible:

```
LLM suggestion (36_llm_recommendations, advisory only)
  → deterministic validation (mapping_backstop_validator statuses)
  → operator review (Review Centre)
  → approval (with scope)
  → persist (rule store + projector into agent-readable sinks)
  → rerun affected stage (orchestrator --resume)
```

LLM output never updates a registry automatically; the projector only runs on
operator approval.

## 7. Trust boundaries and principles

- React app ↔ Operations Control API only. No script invocation, no blob
  access from the browser.
- Operations Control API is the only writer to `operations-control/`.
- The Control Centre never writes into `config/system/fields_registry.yaml`,
  agent code, or `processed-v2` except via the existing promote/publish path,
  and only on explicit approval.
- Operational governance data (workflow runs, approvals, rules, audit) lives in
  its own container, separate from production reporting outputs.
- Every state change is an audit event; the Control Centre can reconstruct any
  workflow, decision, approval, rerun and publication from the container alone.
