# 11 — OCC-owned input readiness (removal of `_READY.json`)

## 1. Dependency map (investigated before changing)

Producers: **none in production code** — the sentinel was only ever created by
humans following the docs, and by test fixtures (`tests/helpers/acquired_pack.py`
and four test files). Consumers, traced with file:line evidence:

| Component | Dependency | Target behaviour | Change |
|---|---|---|---|
| `function_app.py:99-148` (deployed Event Grid entrypoint) | marker branch triggered the whole pipeline; data files were acknowledged only | ALL accepted arrivals register with OCC intake; sentinel arrivals are audited no-ops | **Rewritten** (production file — impact table §6) |
| `apps/blob_trigger_app/router.py:227-240` (`is_marker` gate) + marker-metadata reads (`:259-424`) | the one hard gate + 6 metadata behaviours | unreachable from the production event path; retained for legacy CLI tooling (`ops`, `backfill`, `repin`) pending separately-approved removal | none (legacy-CLI only) |
| `run_records.py:98` / `persistence.py:78` (`is_pack_marker` ledger gate) | non-marker events wrote no run record | OCC workflow store is the run ledger for the new route | none |
| `azure_io.py` pack enumeration ("marker implies complete") | folder listing minus marker = pack | enumeration replaced by per-arrival registration into batches | none (unused on new route) |
| `eventgrid.py`, `path_parser.py`, `file_roles.py`, `schema_fingerprint.py`, pack-key + idempotency logic | marker-agnostic | reused as-is by the new route | none |
| Docs (`trakt_blob_pipeline_runbook.md`, app README, `local.settings.example.json`, `source_registry.example.yaml`) | instructed operators to upload the sentinel | rewritten: source files only; sentinel documented as unsupported legacy | edited |
| Tests referencing the marker (~15 files) | exercise the legacy router path directly | still pass (router unchanged); the new-route contract is covered by `tests/operations_control/test_intake.py` | none |

## 2. Readiness architecture

```
Files uploaded (React OCC) or received (Event Grid blob arrival)
  → registered against tenant/client/portfolio/reporting date (InputBatch)
  → Onboarding Agent recognition: header-first semantic roles
      (loan_extract / property_extract / collateral_extract /
       cashflow_extract / funder_pi_extract), confidence + ambiguity
  → OCC readiness: workflow input requirements
      (config/system/workflow_input_requirements.yaml — administrator-governed,
       system package layer) + effective-configuration status + open decisions
  → immutable internal run manifest (hashes, roles, effective-config hash,
      idempotency key) persisted atomically
  → READY → governed run (auto-start or operator Start)
```

Batch states: `receiving → incomplete | classifying | review_required |
configuration_required → ready → running → completed | failed`. Readiness is
reassessed after **every** registration, recognition result and decision — no
fixed global delay, no sentinel. An ambiguous required role always parks the
batch (`review_required`) with a `file_role` decision in the Review Centre; a
required replacement arriving after start creates a **new batch version**
(`…_v2`) — a running batch's inputs are never mutated.

Contracts: `operations_control/intake.py` (`InputFileRecord` with sha256/size/
recognition fields/duplicate + superseded status; `InputBatch` document with
expected/received/missing roles, blocking decisions, effective-config pinning,
auto-start flag; immutable run manifest). Identity comes from the trusted
execution context (OCC principal or the governed blob path), never from
browser-supplied values alone; every endpoint is tenant-checked server-side.

## 3. Audit evidence

Per transition: `input_batch_created`, `file_registered` (with hash/size/
duplicate/replacement facts), `file_classified` (role/basis/confidence),
`file_classification_overridden`, `readiness_evaluated`, `batch_incomplete`,
`batch_review_required`, `configuration_required`, `batch_ready`,
`run_manifest_created`, `onboarding_started` (with auto/manual flag),
`legacy_sentinel_ignored` — all on the tamper-evident per-client audit chain.

## 4. Automatic vs manual start

`auto_start_when_ready` per batch: the blob-trigger route defaults to **true**
(recurring automated deliveries); the React flow defaults to **false** (the
operator sees "Ready to process → [Start onboarding]"). The readiness decision
is identical in both modes; only the start action differs.

## 5. Deployment & migration

1. Deploy the updated Function App (`function_app.py`) + app setting
   `TRAKT_OPS_CONTAINER=operations-control` (container must exist). Event Grid
   subscription unchanged.
2. From that deployment: source files arriving in `raw-v2` register + start
   automatically; **`_READY.json` uploads are ignored and audited** — inform
   upstream senders they may simply stop uploading it (harmless if they
   continue).
3. In-flight packs mid-upload at cutover: files that arrived before cutover
   were acknowledged-only by the old handler; re-upload (or touch) any one
   pack file after cutover to register the whole pack — Event Grid is not
   retroactive (pre-existing property, unchanged).
4. Legacy CLI tooling (`ops rerun`, `backfill`, `repin`) still uses the old
   router path against historical run records — unchanged, documented as
   legacy in the app README, removal to be proposed separately (it drags ~15
   test files with it).

**Rollback:** redeploy the previous Function App build (`git revert` of
`function_app.py` restores the marker branch verbatim); OCC batches already
created remain as inert governed records. No storage layout, path convention
or Event Grid change is involved — rollback is a code deploy only.

## 6. Production-file change impact (§17 disclosure)

| File | Current behaviour | New behaviour | Deployment impact |
|---|---|---|---|
| `function_app.py` (only production file modified) | data files acknowledged; `_READY.json` triggered enumerate→fingerprint→route→orchestrate inline | every accepted data file → `occ_intake.handle_arrival` (register → recognise → assess → auto-start when complete); sentinel → logged + audited, never triggers | Function App redeploy; needs `TRAKT_OPS_CONTAINER`; behavioural cutover is exactly the sentinel removal described above |
| `apps/blob_trigger_app/occ_intake.py` | — (new file) | Azure-free bridge: path parse → tenant context → batch registration → readiness | additive |

## 7. Known limitations

- Role recognition on the blob route uses approved registry role signatures
  when the source is known; for brand-new sources it is filename+header
  driven, with ambiguity parked to the Review Centre by design.
- The legacy router/CLI path still contains marker code (unreachable from
  production events); its removal is a separately-approved cleanup.
- Multi-portfolio packs register per path segment exactly as before; the
  quiet-period debounce is configurable (`auto_start_quiet_seconds`) but
  defaults to 0 because readiness is role-complete, not time-based.
