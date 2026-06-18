# Phase 2 — Local Snapshot / History Layer

**Status:** Implemented (storage-neutral model + local filesystem adapter +
tests + docs). No Azure, no orchestration, no MI state assembler, no M&A agent,
no risk monitor, no forecast-funded runtime, no Streamlit/chart migration, no
legacy `analytics/` imports, no LLM, no Annex 2/regulatory changes. **Not wired
into the MI Agent runtime.**

**Dependency order:** Phase 0 → Phase 0B → Phase 1 → **Phase 2**.

**Date:** 2026-06-18

MI is *recurring*: it needs weekly/monthly uploads, historical (backfilled)
uploads, and period-to-period comparison. This phase builds the foundational,
storage-neutral snapshot store that persists and retrieves timestamped portfolio
snapshots so future MI states, trends, migrations and forecast-funded analysis
can operate on history.

---

## 1. What was built

A new `snapshot/` package:

| File | Responsibility |
|---|---|
| `snapshot/model.py` | `SnapshotHeader` dataclass, reserved loan-column contracts, the structured-issue model, controlled vocabularies, and header/loan validation (enforces date separation). |
| `snapshot/keys.py` | Deterministic keys: `compute_source_file_id`, `make_snapshot_id`, `normalise_key_part`, `make_pipeline_opportunity_id`, `select_stable_loan_key`. |
| `snapshot/store.py` | Abstract `SnapshotStore` interface + `RegistrationResult`; the temporal resolvers (`resolve_latest` / `resolve_as_of` / `resolve_range` / `resolve_compare`) implemented once on the base in terms of `list_snapshots`. |
| `snapshot/adapters/local_fs.py` | `LocalFsSnapshotStore` — append-only, idempotent, manifest-indexed, latest-pointer-maintaining filesystem adapter (CSV loan rows, JSON header/manifest). |
| `snapshot/adapters/__init__.py`, `snapshot/__init__.py` | Public API surface. |

### 1.1 Snapshot model
`SnapshotHeader` carries: `snapshot_id`, `client_id`, `route`, `cadence`,
`upload_timestamp`, `reporting_date`, `cut_off_date`, `source_file_id`,
`source_file_name`, `row_count`, `schema_version`, `created_at`, `content_hash`,
and free-form `metadata`. All dates are normalised to ISO on construction.

Loan rows stay a pandas DataFrame. Reserved columns are defined as contracts:
the **required** set (`snapshot_id`, `client_id`, `loan_id`, `reporting_date`,
`cut_off_date`, `upload_timestamp`) is always stamped on; the **optional** set
(`source_record_id`, `stable_entity_id`, `opportunity_id`, `portfolio_id`,
`spv_id`, `acquired_portfolio_id`, `origination_date`, `funding_date`,
`acquisition_date`, `spv_transfer_date`, `pipeline_stage`, `funded_status`) is
**not required** — absence yields structured issues, never a crash.

### 1.2 Stable keys
- `compute_source_file_id` → `sha256:<hex>` content hash (storage-portable; not
  an Azure etag).
- `make_snapshot_id` → deterministic `snap_<hash>` from
  `(client, route, reporting_date, source_file_id)`.
- `make_pipeline_opportunity_id` → deterministic `opp_<hash>` for *unfunded*
  pipeline opportunities (legacy SHA1 opportunity-key *idea* re-expressed, not
  copied). Funded `loan_id` (real id) and pipeline `opportunity_id` are kept in
  **distinct namespaces** (`OPP_…`).

### 1.3 Local filesystem adapter
Append-only registration; **idempotent** on identical content (same
`source_file_id` + content hash → returns the existing `snapshot_id`); refuses to
silently overwrite (same id different content, or same client/route/
reporting_date/cadence slot with a different source → `SnapshotConflictError`).
Maintains `manifest.json` for efficient listing and a per-client/route `latest`
pointer.

### 1.4 Strict date separation
`upload_timestamp` (operational) is **never** allowed to become `reporting_date`.
Registration **fails clearly** (`SnapshotValidationError`) when `reporting_date`
is missing, unless `allow_missing_reporting_date=True` is set (for test/backfill,
which records a `missing_reporting_date` issue). `cut_off_date` defaults to
`reporting_date` **only** when `default_cut_off_to_reporting=True`, and records a
`cut_off_date_defaulted_to_reporting_date` issue when it does. All dates stored
ISO.

### 1.5 Structured issues
Plain-dict issues with `code` / `severity` / `message` / `field`. Codes:
`missing_required_header_field`, `missing_reporting_date`, `missing_client_id`,
`missing_source_file_id`, `duplicate_snapshot_same_source`,
`duplicate_snapshot_conflicting_source`, `missing_stable_loan_key`,
`missing_optional_segmentation_field`, `cut_off_date_defaulted_to_reporting_date`,
`invalid_date`, `invalid_route`, `invalid_cadence`. Required header failures
raise; optional data gaps only record issues.

---

## 2. What was intentionally NOT built

- **No Azure Blob adapter**, no Event Grid / function-app ingestion.
- **No orchestration**, no MI state assembler, no M&A agent, no risk monitor,
  no forecast-funded runtime.
- **No snapshot-to-snapshot migration** (that builds on this layer later;
  `analytics_lib.migration` remains a stub).
- **No MI Agent wiring** — this layer stands alone for now.
- **No Streamlit/chart migration, no legacy `analytics/` import, no LLM, no
  Annex 2/XML/regulatory change.**
- Loan rows are CSV (no Parquet) to avoid adding `pyarrow`; the adapter can adopt
  Parquet later without changing the interface.

---

## 3. How this consumes Phase 0B virtual fields

The Phase 0B *virtual* MI semantic dimensions (`portfolio_id`, `spv_id`,
`acquired_portfolio_id`, `reporting_date`, `cut_off_date`, `upload_timestamp`,
`acquisition_date`, `spv_transfer_date`, `pipeline_stage`, `funded_status`) are
exactly the snapshot header fields and reserved loan columns realised here. The
snapshot model is where those virtual dimensions become concrete, per-row data:
header-level dates on the `SnapshotHeader`, and per-loan segmentation/event
columns in the reserved optional-column contract.

---

## 4. How it prepares for Phase 3 (MI states)

Phase 3's state assembler composes this layer with the Phase 1 library:

```
state(selector, filters)
  = SnapshotStore.resolve_*(...)            # this layer: pick snapshot(s)
  → SnapshotStore.load_loans(snapshot_id)   # this layer: loan frame(s)
  → analytics_lib.materialise_buckets/...    # Phase 1: pure analytics
  → stratify / concentration / cohort        # Phase 1
```

The resolvers already provide the temporal selectors states need
(`single` → `resolve_latest`/`resolve_as_of`; `compare` → `resolve_compare`;
`trend` → `resolve_range`). `months_on_book` in Phase 1 already accepts a
reporting/as-of date — which is the snapshot header `reporting_date` this layer
supplies.

---

## 5. How Azure becomes an adapter later

Business logic depends only on the abstract `SnapshotStore`. A future
`AzureBlobSnapshotStore` implements the same four primitives
(`register_snapshot`, `list_snapshots`, `get_snapshot`, `load_loans`) behind the
same interface; the shared resolvers and all callers stay unchanged. The proven
legacy blob *layout/idempotency ideas* can live inside that adapter — never in
the analytics or state logic. No SDK is imported anywhere in this phase.

---

## 6. Integration boundaries (future, not wired now)

- **MI state assembler** will call `SnapshotStore` to resolve one or more
  snapshots, then hand frames to `analytics_lib`.
- **Route contracts** decide whether history is required (`requires_history`):
  MI = true; M&A/Regulatory = false.
- **M&A** may *optionally* read historical snapshots (a small fixed set of
  `as_of` cuts) but does **not** depend on a recurring feed.
- **Regulatory Annex 2** remains strictly point-in-time and does **not** depend
  on `SnapshotStore` at all.

---

## 7. Files changed

- `snapshot/__init__.py`, `model.py`, `keys.py`, `store.py`,
  `adapters/__init__.py`, `adapters/local_fs.py` (new package).
- `tests/test_phase2_snapshot_layer.py` (new, 24 tests).
- `docs/phase2_snapshot_history_layer.md` (this file).

## 8. Tests run

`tests/test_phase2_snapshot_layer.py` — **24 passed**. Combined Phase 0B/1/2
suites — **103 passed**.

## 9. Limitations / deferred items

- CSV loan storage loses dtypes on round-trip (re-inferred on load); a typed
  store (Parquet) is a later, interface-preserving upgrade.
- Idempotency/conflict detection is content-hash based and single-process; no
  cross-process locking (not needed for the local adapter).
- The Azure adapter, MI state assembler, migration engine, and MI Agent wiring
  are explicitly deferred to later phases.
