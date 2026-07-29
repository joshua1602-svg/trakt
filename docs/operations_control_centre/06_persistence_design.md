# 06 — Persistence design

## 1. Principles

- **New dedicated container `operations-control`** — operational governance is
  never mixed with production reporting outputs (`processed-v2`) or pipeline
  state (`trakt-state`).
- **Reuse the existing storage abstraction** — `apps/blob_trigger_app/storage.py`
  (`blob://` URIs, `TRAKT_STORAGE_BACKEND=blob|file`), so local development is
  file-backed with zero code difference. New env var:
  `TRAKT_OPS_CONTAINER=operations-control`.
- **Append-only where it matters** — workflow events, audit, rule versions and
  governed results are immutable once written; "current state" documents are
  small mutable pointers rebuilt from the event history if ever lost.
- **Reconstructable** — the container alone is sufficient to reconstruct every
  workflow, decision, approval, rerun and publication.

## 2. Container layout

```
operations-control/
    {client_id}/
        workflow-runs/
            {workflow_id}/
                workflow.json              # current state pointer (small, mutable)
                events/{seq}_{event}.json  # append-only transition log
                stages/{stage}.json        # current stage state pointers
                orchestrator/              # copy/reference of run_state.json + manifest paths
        governed-results/
            {workflow_id}/{stage}/{result_id}.json   # immutable GAR snapshots
        approvals/
            {decision_id}.json             # review items incl. resolution
        rules/
            {rule_id}/
                current.json               # pointer to active version
                versions/{n}.json          # immutable rule versions (doc 08)
        audit/
            {yyyy-mm-dd}/{seq}_{event_id}.json   # hash-chained audit events
        history/
            {reporting_period}/
                publication.json           # version, approver, rule-version set,
                                           # artefact references, rollback links
    _global/
        rules/…                            # global-scope rules (same shape)
        audit/…                            # cross-client operational audit
        index/
            workflows_open.json            # small denormalised indexes for the
            reviews_open.json              # dashboard/queues (rebuildable)
            publications_pending.json
```

## 3. Document shapes (abridged)

**workflow.json**
```jsonc
{ "workflow_id": "wf_…", "type": "initial_onboarding | new_portfolio | recurring",
  "outcome": "mi | mi_annex2", "client_id": "…", "portfolio_id": "…",
  "reporting_period": "…", "status": "received | running | needs_review | blocked |
  awaiting_publication | published | held | cancelled",
  "stages": {"mapping": {"status": "needs_review", "result_id": "gar_…"}, …},
  "delivery": {"files": [ {"slot": "loan_tape", "name": "…", "content_hash": "sha256:…"} ]},
  "orchestrator_run_id": "…", "created_at": "…", "updated_at": "…" }
```

**events/*.json** — `{seq, event, from, to, actor, at, detail, correlation_id}`
for every transition in the doc 04 state machines.

**audit/*.json** — hash-chained like the remediation ledger
(`record_hash` over a fixed field tuple + `prev_hash`), enabling
tamper-evident export analogous to `export_audit_pack.py`.

**history/{period}/publication.json**
```jsonc
{ "publication_id": "pub_…", "workflow_id": "wf_…", "version": 4,
  "approved_by": "…", "approved_at": "…",
  "rule_versions": {"rule_abc": 3, "rule_def": 1, …},   // exact rule set used
  "artefacts": {"platform_csv": "blob://processed-v2/…", "deck": "…", "xml": "…"},
  "previous_publication_id": "pub_…", "rolled_back_by": null }
```

## 4. Relationship to existing stores (all read-only or via existing writers)

| Store | Ownership | Control Centre access |
|---|---|---|
| `trakt-state/` (run records, approvals, source registry) | `apps/blob_trigger_app` | Read for classification and status; writes only through the existing `approvals.py` / `run_records.py` functions when acting on blob-triggered deliveries |
| `processed-v2/` (production outputs incl. `…/latest`) | pipeline / promote path | Never written directly; publication approval invokes the existing promote/assemble copy path |
| `raw-v2/` | client deliveries | Read; Control-Centre uploads land here in the existing pack layout so the fingerprint/classification machinery applies unchanged |
| Client memory YAML (`{memory_root}/{client_id}/client_memory/…`) | `engine/onboarding_agent/mapping_memory.py` | Written only via the rule projector calling the existing persistence functions |
| `config/system/aliases_pipeline.yaml`, `fields_registry_pipeline.yaml` | `mapping_persistence.py` controlled sinks | Same — projector uses the existing controlled-promotion functions; the core registry is never touched |
| `trakt_exceptions.db` (SQLite) | exception ledger | Phase 2+: read for validation exceptions; remediations continue through its own hash-chained API |

## 5. Retention & recovery

- Nothing in `operations-control/` is deleted; supersede, never overwrite
  (except the small `current`/index pointers, which are rebuildable).
- Recovery drill: rebuild all indexes and `workflow.json` pointers from
  `events/` + `governed-results/`; a maintenance command will exist for this.
- Ops audit export: zip of a client's `audit/` + chain verification, mirroring
  `export_audit_pack.py`.
