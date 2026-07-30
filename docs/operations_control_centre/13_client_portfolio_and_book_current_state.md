# Client, portfolio, asset class and book: where they actually live

Current-state trace behind the governed portfolio selector and the funded /
pipeline wording. Everything below is read from the code, not from labels.

---

## 1. Client and portfolio

### Where clients come from

| Question | Answer |
|---|---|
| Where are clients first created? | Nowhere explicitly. A client comes into existence the first time a workflow is saved for it: `OpsStore.save_workflow` calls `OpsStore.register_client` (`operations_control/stores.py`). |
| Where are clients persisted? | `OpsStore.register_client` → a single index document at `OpsLayout.client_index_uri()` in the operations-control container. |
| What does the Client dropdown read? | `GET /ops/clients` (`operations_control/api/app.py::clients`) = the OCC client index **union** every distinct `client_id` in the source registry, filtered by the operator's tenancy binding. Not mock data, not configuration. |

### Where portfolios come from

| Question | Answer |
|---|---|
| Where are portfolios first created? | The source registry, `apps/blob_trigger_app/source_registry.py::SourceRecord` — one record per `client_id / source_portfolio_id / dataset / frequency`. |
| Where is the client→portfolio relationship persisted? | On that record. `client_id` and `source_portfolio_id` are fields of the same row, so a portfolio belongs to exactly one client by construction. |
| Is the source registry authoritative? | **Yes**, for existence, datasets, frequency and `regime_required`. It is what `occ_intake.outcome_for_source` consults, so it is what the manual route must consult too. |
| Does the configuration administration UI create or edit these? | **No.** `/ops/admin/config/*` administers the system / regime / asset **packages** only. `admin_views.py` never touches `SourceRegistry`. |
| Does onboarding publication register a source? | **Yes.** `OpsEngine._promote_source` runs on publication of a `new_client` / `new_portfolio` workflow and writes the registry record through the existing `apps.blob_trigger_app.approvals` promote path. That is the only automatic writer. |
| Why was Portfolio free text? | Because nothing read the registry on the way in. `CreateBatch.portfolio_id` was a bare `str` and `IntakeService.create_batch` hashed whatever it was given. |
| Could an operator enter a non-existent portfolio? | **Yes**, before this change — and it would open a real, permanently-keyed input pack. |
| What if the portfolio conflicted with the client? | Nothing happened. A pack was created under `client_a/direct_900` even when `direct_900` belonged to `client_b`; there was no cross-client check at all. |

### There is a second, optional portfolio file — and it is not a registry

`config/client/portfolio_registry.yaml` (via `mi_agent/portfolio_metadata.py`,
`TRAKT_PORTFOLIO_REGISTRY`) is an **optional metadata overlay** consumed by the
MI agent: display label, origination capability, forecast treatment, supplied
runoff curve. It does not define which portfolios exist. The OCC catalogue reads
it for the display name **only**, and falls back to the identifier when it is
absent — which is the current ERE state.

## 2. Asset class

**Asset class is not derived from the portfolio.** Traced:

* `ASSET_MODEL` (`operations_control/configuration/packages.py`) declares exactly
  one asset: `equity_release`.
* `EffectiveConfigResolver.resolve` takes `asset_type: str = "equity_release"` as
  a **default parameter** — and no caller anywhere passes it. `OpsEngine.assess_batch`
  and `OpsEngine.start_batch` both call `resolve(...)` without it.

So every delivery resolves to Equity Release because it is the only asset
configured and the resolver defaults to it, not because anything looks it up per
portfolio. The UI therefore shows asset class as a **separate read-only fact**
beside the portfolio and never as the portfolio's identity.

**Gap:** when a second asset is configured, a portfolio→asset mapping will be
needed. The natural home is a field on `SourceRecord`, threaded into
`resolve(asset_type=…)`. Nothing in this change pre-empts that decision.

## 3. `regime_required` → MI-only vs MI + Annex 2

One rule, in one place: `apps/blob_trigger_app/occ_intake.py::outcome_for_source`.

1. Any dataset other than `funded` short-circuits to MI. Pipeline can never carry
   a regime delivery, whatever the registry says.
2. A funded book — direct or acquired — gets `mi_annex2` when any of its funded
   `SourceRecord`s has `regime_required: true`.
3. No record at all → MI, with a warning logged. Regime scope is unknown until
   the book is registered.

The manual route no longer asks the operator for this. `catalogue.describe_portfolio`
derives the same answer from the same records, the form shows it read-only, and
`OpsEngine._require_converged_identity` still refuses a submission where the two
routes would disagree.

## 4. Funded and pipeline: separate governed deliveries

**They are separate, and the architecture leaves no other option.**

| Evidence | Where |
|---|---|
| `dataset` is a path segment, so one blob can only ever be one dataset | `apps/blob_trigger_app/path_parser.py` (7-segment convention) |
| The input pack is keyed on dataset, so funded and pipeline never merge | `IntakeService.deterministic_batch_id` — `dataset` extends the key when it is not `funded` |
| Each pack produces its own manifest, delivery, workflow and audit trail | `OpsEngine.start_batch` → `ensure_manifest` → `register_delivery` → `create_workflow` |
| Pipeline is MI-only by rule, funded may be regime | `occ_intake.outcome_for_source` |
| The OCC refuses the combination at its own door | `OpsEngine.create_batch` → `OPS_DATASET_NOT_REGIME_CAPABLE` |

Uploading both books in one manual workflow is **not** supported, and that is an
architectural decision rather than an implementation gap: the two books have
different reporting scope, different frequencies (ERE delivers funded monthly and
pipeline weekly), and different regime consequences. Merging them would mean one
approval covering two regulatory positions.

### The expected month-end for a client with both

1. **Funded delivery.** Create a manual delivery → the client → the portfolio →
   Funded book. Trakt shows what it will prepare (MI, or MI + ESMA Annex 2 when
   the book is regime-required), takes the period and the files, and opens one
   workflow. Approve and publish it.
2. **Pipeline delivery.** Repeat for the same portfolio and period with Pipeline.
   Trakt always prepares management information only. This is a **separate**
   workflow with its own files, questions, approval and audit trail.

Automated arrivals behave identically — two blob prefixes, two packs, two runs.
The manual route deliberately mirrors that, so an operator's mental model is the
same whichever way files arrive. The UI now says this in one sentence on the book
step rather than leaving it to be inferred.

### Gap: publications are keyed by period, not by dataset

`OpsLayout.publication_uri(client_id, reporting_period)` →
`{client}/history/{period}/publication.json`. Funded and pipeline deliveries for
the same client and period therefore compete for the **same** publication slot:
`_prepare_publication` sees the existing record, bumps `version` and sets
`previous_publication_id`, so the second book publishes as "version 2" of the
first rather than as its own line. Workflows, packs and audit stay correctly
separate; only the publication history conflates them.

Not changed here — it alters publication semantics and needs its own decision on
history and `latest` layout. Flagged as the main remaining data-model gap.

## 5. Terminology found in the UI

| Term | Finding |
|---|---|
| **Portfolio** | Correct, and means `source_portfolio_id` (the source book). It was free text; it is now a governed selection with the display name first and the identifier second. |
| **Asset class** | Was not shown at all. Now shown as its own read-only fact. It is platform-level today, not per-portfolio — see §2. |
| **Book** | Overloaded. It means the *dataset* (funded / pipeline) on the delivery form, and the *book type* (direct / acquired) in the registry and the blob path. Kept as-is for the dataset, and the registry sense is now labelled "Direct originations" / "Acquired book" so the two never appear as the same word. |
| **"Which book are these files for?"** | Shortened to "Which book?", with an explicit note that funded and pipeline are separate deliveries. |
| **"What would you like Trakt to prepare?"** | Now "What will Trakt prepare?" — it is derived, not chosen, and the question form implied otherwise. |
