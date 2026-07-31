# File-by-file implementation ledger

## New — configuration

| File | What it is |
|---|---|
| `config/onboarding/field_catalogue.yaml` | The client-agnostic information model: every field with its scope, who supplies it, its conditional requirement, its validation, its consumers and the artefact it writes into. Contains no client values — a test enforces it. |
| `config/regime/onboarding_standing_fields.yaml` | Which fields of each reporting product are standing rather than delivery-specific, and what each one is unable to represent. Governed as part of the regime package, so a future regime is a configuration change. |

## New — backend

| File | What it does |
|---|---|
| `operations_control/onboarding/catalogue.py` | Loads the catalogue; resolves vocabularies from the modules that own them and regime fields from the governed declaration; evaluates conditional requirements; validates a value against its declared type; derives the client checklist. |
| `operations_control/onboarding/case.py` | The onboarding case: system-generated reference, the nine-status lifecycle with enforced transitions, entities, information requests with evidence, open questions, and its own event history. |
| `operations_control/onboarding/validation.py` | Catalogue-driven validation plus the structural rules no per-field declaration can express — identifier collisions, entity references, duplicate deliveries, regime scope on a non-regime book, a product the portfolios cannot support. |
| `operations_control/onboarding/derivation.py` | What Trakt works out for itself: product eligibility from the asset support model, source registrations from the portfolios, and the expected delivery location from the production storage layout. |
| `operations_control/onboarding/artefacts.py` | Approved answers to Trakt configuration. Catalogue-routed, deterministic, idempotent, merge-not-replace. |
| `operations_control/onboarding/store.py` | Cases under their own reference (a case exists before a client does); immutable configuration versions per client; generated artefacts per version and as current. |
| `operations_control/onboarding/service.py` | The three entry points, answering, information requests, readiness, preview, approval, activation, withdrawal, and the home queues. |
| `operations_control/onboarding/migration.py` | Secondary: reads a legacy client's files into the same generic model, with per-field provenance and the values today's rules refuse surfaced as issues. Writes nothing. |
| `operations_control/api/onboarding_routes.py` | The API. Tenant-bound throughout; approval and activation additionally require an administrator. |
| `scripts/build_mock_catalogue.py` | Generates the browser fixture's copy of the catalogue, so the fixture cannot drift from the governed model. |
| `tests/operations_control/test_onboarding.py` | 77 tests. |

## Changed — backend

| File | Change | Why |
|---|---|---|
| `operations_control/configuration/resolver.py` | `client_config_for()` now imports from `.artefacts`. | The generation module was renamed. |
| `operations_control/configuration/packages.py` | Registered the standing-field declaration in the regime package. | Governed like every other regime file. |
| `operations_control/configuration/admin_views.py` | Added its plain name. | Administrators see a name, not a path. |
| `operations_control/api/app.py` | Mounted the router; registered `CaseError` so an illegal transition returns an operator-safe envelope rather than a 500. | One API, one auth model, one error contract. |

## Removed

| File | Why |
|---|---|
| `operations_control/onboarding/model.py` | Replaced by `case.py` + `catalogue.py` + `validation.py`. Its single-entity, two-status model was the core of what the correction rejected. |
| `operations_control/onboarding/generation.py` | Replaced by `artefacts.py`, which routes through the catalogue instead of hard-coding each field, and no longer seeds from any client's file. |

## New — frontend

| File | What it does |
|---|---|
| `src/api/onboardingTypes.ts` | Wire types for the case, catalogue, requests and preview. |
| `src/api/mockCatalogue.ts` | **Generated** from the governed YAML. |
| `src/api/MockOnboarding.ts` | In-memory onboarding for demonstration and browser tests. Carries no real client; the migration fixture is a synthetic one. |
| `src/components/onboarding/CatalogueForm.tsx` | Renders whatever the catalogue declares. No hard-coded field lists anywhere in it. |
| `src/screens/onboarding/Home.tsx` | Queues, active clients, and a blank-start primary action. |
| `src/screens/onboarding/CaseWizard.tsx` | The nine-step case: catalogue-driven steps, the reporting and regime steps, deliveries, the client checklist, readiness, preview, approve, activate. |
| `src/screens/onboarding/ClientView.tsx` | An active client: general, entities, portfolios, reporting, deliveries, history, cases. |
| `src/screens/onboarding/Onboarding.test.tsx` | 22 browser tests, including two at 390px. |

## Removed — frontend

| File | Why |
|---|---|
| `src/screens/onboarding/Wizard.tsx` | Hard-coded its own field lists and branched on `origin` to decide whether the page was an adoption. |
| `src/screens/onboarding/ClientEditor.tsx` | Its primary action doubled as the adoption entry. |

## API surface

| Route | Purpose | Access |
|---|---|---|
| `GET /ops/onboarding/reference` | The information model the wizard renders | Operator |
| `GET /ops/onboarding/home` | The working queues and active clients | Operator |
| `POST /ops/onboarding/cases` | **Start a new client — blank** | Operator |
| `POST /ops/onboarding/cases/migration` | Bring in a legacy client | Operator, tenant-bound |
| `POST /ops/onboarding/cases/amendment` | Change an active client | Operator, tenant-bound |
| `GET|PUT /ops/onboarding/cases/{id}` | Read / answer a step | Operator |
| `POST /ops/onboarding/cases/{id}/pipeline-book` | Expect a pipeline book too | Operator |
| `DELETE /ops/onboarding/cases/{id}/sources/{portfolio}/{dataset}` | Remove a delivery | Operator |
| `GET /ops/onboarding/cases/{id}/checklist` | What the client still owes | Operator |
| `POST /ops/onboarding/cases/{id}/requests` | Raise an information request | Operator |
| `POST …/requests/{id}/sent` · `/response` · `/review` | Track it | Operator |
| `POST /ops/onboarding/cases/{id}/questions` · `/resolve` | Open questions | Operator |
| `GET /ops/onboarding/cases/{id}/preview` | Exactly what activation would write | Operator |
| `POST /ops/onboarding/cases/{id}/submit` · `/changes-required` | Move it along | Operator |
| `POST /ops/onboarding/cases/{id}/approve` | Record the decision | **Administrator** |
| `POST /ops/onboarding/cases/{id}/activate` | **Write the configuration** | **Administrator** |
| `POST /ops/onboarding/cases/{id}/withdraw` | End it, writing nothing | Operator |
| `GET /ops/onboarding/clients/{id}` · `/versions/{n}` | An active client and its history | Operator, tenant-bound |

## Storage

```
_onboarding/cases/ONB-2026-0001.json          a case, before it has a client
_onboarding/sequence-2026.json                the reference counter
_onboarding/clients.json                      which clients are active
{client}/onboarding/versions/0001.json        immutable version
{client}/onboarding/current.json              pointer
{client}/onboarding/artefacts/0001/…          generated artefacts, per version
{client}/onboarding/artefacts/current/…       what the resolver reads
```

Source registrations go to the durable source registry. There is one source
registry and onboarding writes to it.
