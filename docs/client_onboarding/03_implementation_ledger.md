# File-by-file implementation ledger

## New — backend

| File | What it does |
|---|---|
| `config/regime/onboarding_standing_fields.yaml` | The governed declaration of which fields of each reporting product are standing, and which existing artefact each answer is written into (`writes_to:`). Defines no new fields. Registered in the regime package, so a future regime is a configuration change. |
| `operations_control/onboarding/__init__.py` | Package surface and the statement of intent: generate the existing artefacts, never a parallel set. |
| `operations_control/onboarding/model.py` | The standing-configuration model (one dataclass per step), governed vocabulary resolution from existing owners, operator-safe validation, and the two derivations — reporting eligibility and source registrations. |
| `operations_control/onboarding/store.py` | Immutable versioned profiles, per client. New version on every approval; the prior version's content is never rewritten. Also stores generated artefacts per version and as current. |
| `operations_control/onboarding/migration.py` | Adopts an existing client: reads the client YAML, Annex 12 overlay, portfolio registry, tenancy file and source registry, returns a populated profile plus per-field provenance, the gaps the legacy files cannot answer, and the base documents so unmanaged blocks survive. Writes nothing. |
| `operations_control/onboarding/generation.py` | Profile → governed artefacts. Merge, never replace. `plan()` shows what would change; `apply()` writes only after approval. Preserves an existing source record's approved mapping and pinned fingerprint. |
| `operations_control/onboarding/service.py` | The workflow: draft → answer → review → approve. Field-level before/after diffing, audit on every approval, and the client/version views. |
| `operations_control/api/onboarding_routes.py` | The API. Tenant-bound on every route; approval additionally requires an administrator. |
| `tests/operations_control/test_onboarding.py` | 58 tests across vocabularies, validation, derivation, adoption, generation, the workflow, versioning and audit, the client views, the resolver seam, and API tenancy. |

## Changed — backend

| File | Change | Why |
|---|---|---|
| `operations_control/configuration/resolver.py` | Added `client_config_for(client_id)`; `resolve()` now uses it instead of the fixed path. | Makes the generated configuration the one deliveries actually run with. A client without a profile still resolves the repository file exactly as before, so this shipping does not adopt anyone. |
| `operations_control/configuration/packages.py` | Added `config/regime/onboarding_standing_fields.yaml` to `LAYER_FILES[LAYER_REGIME]`. | The standing-field declaration is governed like every other regime file. |
| `operations_control/configuration/admin_views.py` | Added its plain-English label. | Administrators see a name, not a path. |
| `operations_control/api/app.py` | Mounted the onboarding router. | One API, one auth model. |

## New — frontend

| File | What it does |
|---|---|
| `src/api/onboardingTypes.ts` | The wire types, mirroring the backend model. |
| `src/api/MockOnboarding.ts` | In-memory onboarding for demonstration and browser tests, seeded with ERE as it exists today. A fixture, not a second implementation. |
| `src/components/onboarding/primitives.tsx` | Shared form and presentation pieces, including the action chip that reads "will be created" at review and "created" in history. |
| `src/screens/onboarding/Home.tsx` | Client Onboarding home: onboarded and not-yet-onboarded clients, with adoption offered. |
| `src/screens/onboarding/Wizard.tsx` | The seven-step wizard, including the review and approval screen. |
| `src/screens/onboarding/ClientEditor.tsx` | The existing-client editor: General / Portfolios / Reporting / Regimes / Source registrations / History. |
| `src/screens/onboarding/Onboarding.test.tsx` | 13 browser tests over the three screens. |

## Changed — frontend

| File | Change |
|---|---|
| `src/api/OpsClient.ts` | Twelve onboarding methods on the client contract. |
| `src/api/HttpOpsClient.ts` | Their HTTP implementation. |
| `src/api/MockOpsClient.ts` | Their mock implementation, delegating to `MockOnboarding`. |
| `src/App.tsx` | Three routes under `/onboarding`. |
| `src/components/Shell.tsx` | The Client Onboarding navigation entry. |
| `src/lib/copy.ts` | Its user-facing strings (a test enforces plain English). |

## Deliberately unchanged

- `config/client/config_client_ERM_UK.yaml` and every other repository client
  file. They remain the seed and the fallback.
- `apps/blob_trigger_app/source_registry.py`. Onboarding writes through the
  existing `upsert`.
- `operations_control/engine.py`, `intake.py`, `annex2/`. The delivery workflow
  is untouched.
- `config/tenancy.yaml`. Authorisation, not business configuration.

## API surface

| Route | Purpose | Access |
|---|---|---|
| `GET /ops/onboarding/vocabularies` | Governed option lists and the standing-field declaration | Operator |
| `GET /ops/onboarding/clients` | Home list, tenant-scoped | Operator |
| `GET /ops/onboarding/clients/{id}` | The client editor's six tabs | Operator, tenant-bound |
| `GET /ops/onboarding/clients/{id}/versions/{n}` | One historical version | Operator, tenant-bound |
| `POST /ops/onboarding/drafts` | Open a draft, blank or adopted | Operator |
| `GET /ops/onboarding/drafts/{id}` | Resume a draft | Operator |
| `PUT /ops/onboarding/drafts/{id}` | Save one step | Operator |
| `GET /ops/onboarding/drafts/{id}/review` | Exactly what approval would write | Operator |
| `POST /ops/onboarding/drafts/{id}/approve` | Write it | **Administrator** |
| `POST /ops/onboarding/drafts/{id}/discard` | Abandon a draft | Operator |

## Storage

Everything lives in the existing `operations-control` container, under the
client's own prefix:

```
{client}/onboarding/versions/0001.json      immutable version
{client}/onboarding/current.json            pointer
{client}/onboarding/drafts/{id}.json        work in progress
{client}/onboarding/artefacts/0001/…        generated artefacts, per version
{client}/onboarding/artefacts/current/…     what the resolver reads
_index/onboarded_clients.json               index
```

Source registrations go to the durable source registry, not here — there is one
source registry and onboarding writes to it.
