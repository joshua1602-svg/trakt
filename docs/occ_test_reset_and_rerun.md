# Discarding the pre-OCC test run and reprocessing both books

The canonical artifacts currently in Azure came from a disposable test run
completed **before the OCC workflow was operational**. They are not a governance
record, are not versioned, and are not to be retained. This runbook removes them
and reprocesses the direct and acquired books through the current OCC workflow,
so the mapping decisions the model proposes and the operator approves are the
decisions production actually executes.

Nothing here is executed by Trakt. Every deletion is a manual operator action.

> **Before you start.** Confirm you hold the source files for re-upload. Step 5
> deletes the raw copies, and Trakt does not keep another.

---

## 1. Identity of the run being discarded

Everything below is derived from the registered sources
(`config/source_registry.example.yaml`) and the blob path convention
(`apps/blob_trigger_app/path_parser.py`).

| | Direct book | Acquired book |
|---|---|---|
| `client_id` | `ERE` | `ERE` |
| `source_portfolio_id` | `direct_001` | `acquired_001` |
| `source_book_type` | `direct` | `acquired` |
| `dataset` | `funded` | `funded` |
| `frequency` | `monthly` | `ad_hoc` |
| `pack_key` | `ERE_direct_001_funded_monthly_{period}` | `ERE_acquired_001_funded_ad_hoc_{period}` |

`{period}` is the reporting-period folder of the test run (e.g. `2026-06-30`).
Read it off the raw path before deleting anything, and substitute it everywhere
below. **Confirm the frequency segment on the real blobs** — a book registered at
one frequency may have been delivered under another.

---

## 2. Prefixes to delete

Delete the **contents** at these prefixes. Do **not** delete the containers.

### `raw-v2` — raw source copies

```
raw-v2/ERE/direct/funded/monthly/direct_001/{period}/
raw-v2/ERE/acquired/funded/ad_hoc/acquired_001/{period}/
```

Delete these **only once you have confirmed the source files are available for
re-upload** (step 5 of the rerun). Everything else can be recreated by Trakt;
these cannot.

### `inbound` — legacy landing container

No current code path writes to `inbound`; the Event Grid handler
(`function_app.py`) watches `TRAKT_BLOB_CONTAINER`, which is `raw-v2` in
production. Anything under `inbound` is therefore from the legacy flow. Because
no module in this repository defines its layout, **enumerate the container and
delete only the blobs whose path or name identifies this client and these two
portfolios** (`ERE`, `direct_001`, `acquired_001`, and the two source
filenames). Leave anything you cannot attribute to this test run.

### `outbound` — legacy pipeline outputs

Same situation: the current handler does not write here, so the contents are the
legacy test run's outputs. Enumerate and delete **only** the blobs attributable
to `ERE` / `direct_001` / `acquired_001`.

### `processed-v2` — component and platform canonicals

```
processed-v2/accepted/ERE/direct_001_canonical_typed.csv
processed-v2/accepted/ERE/acquired_001_canonical_typed.csv
processed-v2/platform/ERE/latest/
processed-v2/platform/ERE/{period}/
processed-v2/regime/ERE/{period}/
processed-v2/mi/ERE/
```

The two `accepted/` files are the component canonicals; `platform/ERE/latest/`
is what MI reads. Both must go, or the Assembler will combine a fresh component
with a stale one — it selects the latest snapshot **per portfolio**, and a
portfolio with only an old snapshot keeps it.

Also remove, if the test run produced them:

```
processed-v2/decks/ERE/
processed-v2/pipeline/ERE/
```

### `trakt-state` — run records and idempotency

```
trakt-state/runs/ERE_direct_001_funded_monthly_{period}.json
trakt-state/runs/ERE_direct_001_funded_monthly_{period}/
trakt-state/runs/ERE_acquired_001_funded_ad_hoc_{period}.json
trakt-state/runs/ERE_acquired_001_funded_ad_hoc_{period}/
```

The `.json` is the run record; the directory of the same name holds that pack's
gate diagnostics, LLM recommendations, governance artifacts and onboarding
decision copies. Delete both, or a re-upload is recognised as an already-processed
pack.

Then, in the same container:

* `trakt-state/registry/source_registry.yaml` — **edit, do not delete.** For the
  two records above, clear `expected_schema_fingerprint`, `expected_columns`,
  `approved_mapping_id`, `mapping_config_path`, `last_successful_run_id` and
  `last_successful_reporting_period`, and set `status: pending_review`. A pinned
  fingerprint routes the upload `deterministic`, which skips the model proposal
  and operator approval this rerun exists to perform. Leave every other client's
  record untouched.
* `trakt-state/approvals/` and `trakt-state/events/` — these are keyed by
  approval and event id, not by client, so there is no client prefix to delete.
  Open each blob and remove only those whose `pack_key` or `blob_path` names one
  of the two packs above.

### `operations-control` — workflow and mapping artifacts

```
operations-control/ERE/workflow-runs/
operations-control/ERE/deliveries/
operations-control/ERE/decisions/
operations-control/ERE/approved-decisions/direct_001/funded/
operations-control/ERE/approved-decisions/acquired_001/funded/
operations-control/ERE/audit-keys/
operations-control/ERE/history/
```

`approved-decisions/{portfolio}/{dataset}/` holds the standing mapping contract
(`34_approved_decisions.yaml`, `12_approved_mapping_overrides.yaml`). It must go:
while it exists the next delivery is treated as a known source and applies the
old contract instead of asking again.

Two judgement calls, both yours:

* `operations-control/ERE/rules/` — the versioned approved rule store. Deleting
  it discards the approvals made during the test run so the fresh run starts
  from the model's proposals with nothing inherited. That is the intent of a
  clean rerun, but it is a deliberate discard of a governance record: decide
  explicitly. **Never touch `operations-control/_global_/rules/`** — those are
  asset- and global-scope rules belonging to every client.
* `operations-control/ERE/audit/` — the append-only audit chain, including its
  `_head.json`. Deleting it removes the record that the test run happened.
  Retaining it is the safer choice and does not affect the rerun.

`operations-control/ERE/mi-queries/` holds MI question telemetry and has no
effect on the rerun; clear it only if you want the question history reset.

### `operations-control-synthetic`

**Leave entirely untouched.**

---

## 3. Rerun sequence

1. **Deploy the corrected code and configuration.** The MI production path now
   consumes the OCC-approved mapping contract as tier-0 mapping authority, and
   `config/client/config_client_ERM_UK.yaml` requires one for `direct_001` and
   `acquired_001`.
2. **Confirm the legacy blobs and state are gone**, and that the two source
   registry records read `status: pending_review` with no pinned fingerprint.
3. **Upload the direct source book** to
   `raw-v2/ERE/direct/funded/monthly/direct_001/{period}/`.
4. **Complete model proposal and operator approval in OCC** for the direct book.
   The four source concepts are separate columns and each takes its own target;
   `Loan Type`, `Product Category`, `Lump Sum or Drawdown` and `Policy Status`
   must not share one.
5. **Upload the acquired source book** to
   `raw-v2/ERE/acquired/funded/ad_hoc/acquired_001/{period}/`.
6. **Complete model proposal and operator approval in OCC** for the acquired book.
7. **Confirm both approved mapping artifacts exist** at
   `operations-control/ERE/approved-decisions/direct_001/funded/12_approved_mapping_overrides.yaml`
   and the `acquired_001` equivalent, and that each carries a `contract_scope`
   naming its own client, portfolio and dataset.
8. **Run the MI pipeline** for both portfolios.
9. **Confirm the mapping receipts show `operator_approved`.** In each run's
   `{stem}_header_mapping_report.json`, `columns_decided_by_operator_approval`
   lists the approved columns and each of those rows reads
   `mapping_method: operator_approved` with its `approved_rule_id` and
   `approved_rule_version`. A row that names any other method was decided by
   automated matching, not by the contract.
10. **Validate the new component and combined canonicals** against the
    acceptance list below.
11. **Rerun the original MI question.**

---

## 4. What the operator is checking at step 10

Against the real files, not this repository:

* 73 direct rows, 885 acquired rows, 958 unique combined loans;
* no lost or duplicated `platform_loan_key`;
* the four direct product concepts each in their own canonical field, still
  distinct;
* direct `account_status = A`;
* acquired `youngest_borrower_age` populated wherever a borrower date of birth
  is usable, null and flagged for review where none is;
* acquired `broker_channel = Acquired_001` on every row with no source value,
  and the source value retained wherever there is one;
* the platform canonical built from the fresh OCC-run components only —
  check `platform_canonical_manifest.json`'s `portfolios[].selected_canonical_path`
  and `input_file_hash`;
* the original MI question answers.

None of these figures was reproduced in the repository; the tests prove the
mechanism on synthetic data only.

---

## 5. Known follow-ups, deliberately not in this change

* **Stable applicable schema.** Gate 1's default `active` schema is
  `core_required | mapped_fields`, so a canonical field applicable to the asset
  class but unmapped in one tape is absent from that component rather than
  present and null. The acquired derivation and the portfolio default close the
  two fields this book needs; the general fix does not ship here.
* **Capability coverage.** `trakt_core.capability.PortfolioShape.has()` gates on
  column presence, not on non-null coverage, so a present-but-empty column reads
  as available. Related to the above and out of scope for the same reason.
* **Global alias migration.** `"Lump Sum or Drawdown"` and `"product category"`
  resolve through `config/system/aliases_analytics.yaml`, which is the global
  library. An approved contract now outranks them for the books that have one,
  but the entries still widen every other client's matching surface. Moving them
  to a client overlay is a behaviour change for existing clients and needs its
  own approved migration.
