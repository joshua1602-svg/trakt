# Manual delivery, configuration readability, and the delivery workflow

Implementation note for the three changes made to the Operations Control Centre.
Section 1 is the current-state finding that the other two were built on top of.

---

## 1. Current state of the "Start something new" route

Traced end to end from `frontend/operations-control-ui/src/screens/NewWorkflow.tsx`
through `HttpOpsClient` to `operations_control/api/app.py` and the engine.

### What it did

| Question | Answer before this change |
|---|---|
| Creates a governed `raw-v2` prefix? | **No.** Nothing was written to the raw container at all. |
| Uploads selected files into `raw-v2`? | **No.** There was no upload of any kind. The form had a free-text "Where are the files?" box. |
| Destination derived server-side from controlled fields? | **No.** The browser sent an arbitrary filesystem path as `{"path": …}` to `POST /ops/batches/{id}/files`. |
| Creates or uploads `_READY.json`? | **No** — and deliberately so. See "The `_READY.json` question" below. |
| `_READY.json` written last? | Not applicable. |
| Triggers the Event Grid / Function App pipeline? | **No.** Files never reached the watched container, so no blob-created event fired. |
| Creates an OCC batch and workflow? | Batch **yes**; workflow **only** if the operator later pressed "Start onboarding" on the input-pack screen. |
| Redirects the user to the resulting workflow? | **No.** It navigated to `/batches/{id}`. |
| Only metadata / a UI shell? | **Neither.** It created real governed state through the real intake service — but it could not receive a file. |
| Same path or period already exists? | Handled well already: `IntakeService.deterministic_batch_id` keys the pack on client + portfolio + period + outcome (+ dataset when not funded), so a repeat returns the existing open pack; a pack that has already started opens a `_v2` successor instead of being mutated. |
| Authentication, tenancy, path validation? | Auth and tenancy: enforced (`authenticate`, `require_client`, 404 not 403). Path validation: **none whatsoever**. |

### Exact code involved

| Step | Location |
|---|---|
| Form | `frontend/operations-control-ui/src/screens/NewWorkflow.tsx` |
| Create pack | `POST /ops/batches` → `app.create_batch` → `OpsEngine.create_batch` → `IntakeService.create_batch` |
| "Add the files" | `POST /ops/batches/{id}/files` → `app.register_batch_file` → `OpsEngine.register_batch_file` → `IntakeService.register_file` |
| Readiness | `OpsEngine.assess_batch` → `IntakeService.classify` / `IntakeService.assess` |
| Start | `POST /ops/batches/{id}/start` → `OpsEngine.start_batch` → `IntakeService.ensure_manifest` → `register_delivery` → `create_workflow` |
| Automated equivalent | `function_app.py` → `apps/blob_trigger_app/occ_intake.handle_arrival` → the same `OpsEngine` methods |

### Was it production-ready?

**No.** Two findings, one of them a security defect:

1. **It could not be used.** `OpsEngine.register_batch_file` required
   `Path(source_path).exists()` — a path on the **API server's own filesystem**.
   A browser user has no way to put a file there. The route was only ever
   usable from a developer machine where the server and the files shared a disk.
2. **Client-controlled server path (security).** The endpoint passed the
   browser's string straight to `Path(...).read_bytes()` with no validation, no
   allow-list and no traversal check. Any authenticated operator could register
   any file readable by the API process — another tenant's staged inputs,
   `/etc/passwd`, anything — into their own batch, and then read it back through
   the pack screen. This was the most serious issue found.

Lesser risks: no redirect to the resulting workflow, and no bound on what was read.

### The `_READY.json` question

The brief assumes a `_READY.json` sentinel written last. **This repository has
deliberately removed that mechanism.** `operations_control/intake.py` names it
`LEGACY_SENTINEL`, records any arrival of it as
`legacy_sentinel_ignored`, and never lets it trigger anything;
`apps/blob_trigger_app/occ_intake.py` and the root `function_app.py` docstring
say the same. Readiness is now owned by the OCC (recognised input roles +
effective configuration + open decisions), and its governed artefact is the
**immutable internal run manifest** written by `IntakeService.ensure_manifest`.

Re-introducing `_READY.json` would have created exactly the second ingestion
path the brief forbids. So the requirement was honoured against the artefact
that actually authorises processing today: **the run manifest is written last,
after every source file is placed and registered.** That ordering is asserted
by `tests/operations_control/test_manual_delivery.py::TestUploadOrdering`.

### What was implemented

The narrowest production-safe version of the intended flow:

1. `POST /ops/batches/{batch_id}/upload` — a multipart route that carries file
   **content** only.
2. `operations_control/manual_intake.py` derives the destination from the pack's
   own controlled fields, then **re-parses it with the production path parser**
   (`apps/blob_trigger_app/path_parser.parse_blob_path`). If the automated route
   would not accept the location, nothing is written (fail closed).
3. Filenames are reduced to a safe leaf name; traversal, control characters,
   unreadable file types and `_READY.json` itself are refused **before** any
   byte is written, so a rejected upload leaves no half-pack behind.
4. Every file is placed at its governed location and registered through the
   **existing** `IntakeService.register_file` — the same call the blob trigger
   makes — before readiness is assessed.
5. Readiness, the run manifest, the delivery and the workflow are all produced
   by the existing intake path. No business logic was duplicated.
6. The screen redirects to the resulting workflow.

### Both doors must agree on the delivery

A manual delivery is filed where an automated one would be, so the Event Grid
trigger sees it too. If the two routes identified that location as *different*
deliveries, the same files would split across two input packs — each incomplete,
neither publishable.

`OpsEngine.automated_identity` replays the automated derivation on the exact
location about to be written: the production path parser reads the location
back, and `occ_intake.outcome_for_source` — the trigger's own workflow rule,
promoted to a public name so there is one rule in one place — chooses the
workflow. `_require_converged_identity` refuses the upload unless the resulting
`deterministic_batch_id` is the pack the operator is uploading into. The check
runs **before the first byte is written**, and the refusal is audited
(`manual_delivery_refused`).

In practice this catches the one real divergence: an operator asking for the
ESMA Annex 2 delivery on a book the source registry does not flag as
regime-required (or the reverse). The operator is told which one the automated
route would prepare, in plain language, and nothing is left half-written.

The old free-text path route is kept for server-side tooling but is now
**fail-closed**: administrator-only, and refused unless the location resolves
inside a directory an administrator has allow-listed
(`TRAKT_OPS_SERVER_PATH_ROOTS`, empty by default). The browser client no longer
has a method to call it.

### Naming

The automated Event Grid route is the normal way deliveries arrive; this is the
manual door. The page is therefore **"Create a manual delivery"**, with the
guidance sentence about ad hoc, backdated and replacement deliveries — and it
uses the six guided steps, because it is a real end-to-end intake path.

---

## 2. Platform Configuration

`describe_readiness` / `with_readiness` in
`operations_control/configuration/admin_views.py` produce the operational answer
on the server, in plain language: readiness state, whether the package can
process a delivery, and the warnings that would affect processing (including the
retained ESMA Annex 2 draft-schema warning).

`ConfigAssets.tsx` leads with that answer under the heading **"Can this
configuration safely process a delivery?"**. Package identifiers, groupings,
dependencies, profile versions and issue-policy counts moved behind
**"Administration and technical details"** on the card; version history, parts,
impact, comparison, drafts and rollback moved behind the same disclosure at page
level. Nothing was removed, and the detailed heading is now named for the asset
("Equity Release configuration").

## 3. The delivery workflow

`operations_control/api/workflow_view.py` assembles a durable seven-step case
file from documents the engine already owns — the run, its governed agent
results, the input batch, the decisions and the publication. It decides nothing;
it reads. `GET /ops/workflows/{id}` now carries `steps` and `current_step`.

`WorkflowDetail.tsx` renders it as a persistent case file: every step visible,
completed steps expandable, the page opening on the step the delivery is at.
The approval decision lives inside the workflow with its evidence summary above
it, its scope question below it, and a plain consequence sentence before
confirmation.

Approval scope is `delivery | portfolio | client`
(`contracts.PUBLICATION_SCOPES`); the platform-wide scope is absent from the
screen **and** refused by `OpsEngine.approve_publication`.

**The scope is recorded, not enforced.** It is written to the publication and
the hash-chained audit trail, and nothing reads it back to approve a later
delivery. The wording says exactly that — "Trakt records your answer so it is on
the delivery's record. It does not publish anything on its own — every delivery
is approved by a person." A test asserts that no option explanation or
consequence sentence promises automatic future approval, so the wording cannot
drift ahead of the behaviour. If enforcement is added later, that test is where
the promise should be updated.

A queue item now opens the delivery it belongs to. `/reviews/{id}` still works:
it forwards to the workflow when the question belongs to one, and answers the
question in place when it was raised against an input pack that has no workflow
yet (a file still to be identified).
