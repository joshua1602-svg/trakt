# 09 — Migration impact assessment & approval checkpoint

## 1. Summary

The design requires **zero modifications to existing pipeline components** for
its core. Everything is additive: two new top-level trees, one new blob
container, new environment variables, and read-only/adapter integration with
existing seams. Three small optional integration points are flagged below;
each would follow the mandated change-approval procedure individually and none
is required for Phase 2 to begin.

## 2. Files requiring modification (existing code)

**None required.** The non-invasive strategy relies on:
- subclassing `engine.orchestrator_agent.adapters.AgentAdapters` (public seam),
- read-only consumption of numbered manifests, `run_state.json`, run records,
- existing writer functions (`approvals.py`, `mapping_memory.py`,
  `mapping_persistence.py`) called as libraries with their own guarantees.

### Optional integration points (each needs separate approval; not in scope now)

| # | Candidate change | Why it might be wanted | Without it |
|---|---|---|---|
| O1 | `apps/blob_trigger_app/router.py`: env-gated flag to register arriving packs as Control Centre workflow runs instead of/alongside auto-processing | Blob-arriving deliveries appear on the dashboard automatically | Control Centre discovers arrivals by reading `trakt-state/runs` + approvals read-only — dashboard still works, slightly less real-time |
| O2 | `engine/orchestrator_agent/orchestrator.py`: accept externally-supplied adapters if any hook proves non-overridable | Cleaner injection | Current `AgentAdapters` surface already supports injection (`adapters` parameter exists); likely unnecessary |
| O3 | CI/deploy workflow additions (new GitHub Actions for ops API + ops UI) | Deployment | Manual deploy initially |

## 3. New files

```
operations_control/                        # new Python package (doc 01 §3.1)
    contracts/  engine/  adapters/  stores/  rules/  api/
frontend/operations-control-ui/            # new React app (doc 02)
docs/operations_control_centre/            # this design pack
deploy/trakt-ops-api/                      # Dockerfile + provisioning (mirrors deploy/trakt-mi-api)
tests/operations_control/                  # contract, state-machine, adapter, language-rule tests
```

Plus a new Azure Blob container `operations-control` (infrastructure, not code)
and new env vars: `TRAKT_OPS_CONTAINER`, `OPS_API_*` (auth/CORS mirroring
`MI_AGENT_*`), `VITE_OPS_API_URL`.

## 4. Deleted files

**None.** Existing Streamlit review surfaces (`ui/onboarding_review.py`,
`exception_queue.py`, `engine/*/streamlit_*`) and
`mi_agent_operator/static/operator_ui.html` remain untouched and functional;
they are candidates for deprecation only after the Control Centre reaches
parity, as a separately approved step.

## 5. Upstream impact (things that feed the new layer)

- Client deliveries: unchanged. Uploads via the Control Centre land in the
  existing `raw-v2` pack layout, so fingerprinting/classification behave
  identically to a direct blob upload.
- Blob trigger (`function_app.py`): unchanged; continues to operate. Until O1
  is approved, both paths coexist — trigger-processed runs are visible
  read-only in the Control Centre.
- Existing agent CLIs and Streamlit tools: unchanged and usable in parallel.

## 6. Downstream impact (things the new layer feeds)

- `processed-v2/**/latest`: written only via the **existing** promote/assemble
  path, now behind an explicit approval. Control-Centre-initiated runs stage
  outputs in run-scoped directories until publication is approved.
- MI API / dashboards / decks: unchanged — they read the same
  `processed-v2` locations with the same shapes.
- Client memory and pipeline alias files: written only through existing
  controlled functions; agents observe the same file contracts as today.

## 7. Pipeline impact

- Business calculations: none — no agent logic touched.
- Ordering/gating: unchanged — the orchestrator's own readiness ladder remains
  the enforcement; the Control Centre only reads it and decides when to rerun.
- Performance: agents run as they do today; the new layer adds manifest reads
  and JSON writes (negligible).
- Behavioural change visible to anyone today: **only** that
  Control-Centre-initiated runs do not auto-publish. Blob-triggered runs keep
  their current behaviour until O1 is separately approved.

## 8. Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Adapter drifts from agent manifest shapes over time | Medium | Status shown wrongly in UI | Contract tests pinning each manifest's consumed fields; adapters fail closed to `blocked` with a support reference, never guess |
| Dual write surfaces (ops UI + legacy Streamlit/CLI) make conflicting decisions | Medium | Confusing rule state | Both paths converge on the same underlying artefacts; Control Centre re-reads before write and surfaces "changed since your approval" conflicts |
| In-process background runs die with the API process | Medium | Stuck `running` workflows | Runs are resumable by design (orchestrator `--resume` + persisted state); engine marks stale runs and offers one-click resume |
| Rule projection bug corrupts client memory | Low | Wrong auto-mappings | Projector only calls existing persistence functions, is idempotent, and every write is versioned in `operations-control` for replay; client memory files are small YAML, recoverable from rule store |
| Non-technical copy drifts into technical leakage | Medium | UX principle broken | Forbidden-vocabulary contract tests over rendered strings (doc 07 §4) |
| New container misconfigured in prod | Low | Ops layer down, pipeline unaffected | Fail-closed startup check in `/health`; pipeline has zero dependency on the new container |

## 9. Rollback strategy

The layer is additive, so rollback is removal, not restoration:

1. Stop/deprovision the ops API and UI (own deployments; nothing else depends
   on them).
2. Existing pipeline, blob trigger, MI API, dashboards continue untouched —
   they never depended on `operations-control/`.
3. Rules already projected into client memory / pipeline aliases were written
   through the same functions operators use today and remain valid,
   individually reviewable artefacts; they can be kept or reverted from the
   versioned rule store.
4. The `operations-control` container can be retained (audit history) or
   archived; nothing reads it except the ops layer.
5. If O1 was enabled, disable its env flag — the trigger reverts to current
   behaviour instantly.

## 10. Approval checkpoint

Phase 1 (this design pack) contains **no implementation**. Approval is
requested to proceed to Phase 2:

1. Scaffold `operations_control/` (contracts, stores, engine skeleton, API
   skeleton with auth) + tests.
2. Scaffold `frontend/operations-control-ui/` (shell, Dashboard, Workflow
   screen against the Mock client).
3. Governed adapters + manifest readers for onboarding → delivery.
4. Review Centre + rule store + projector (client scope first).
5. Publication gating + history.

No existing file is modified in Phase 2. Optional integrations O1–O3 will each
be brought back for explicit approval with the full impact procedure before
any change is made.
