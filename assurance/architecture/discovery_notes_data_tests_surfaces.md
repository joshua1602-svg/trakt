# Discovery notes — data, fixtures, tests, delivery surfaces

Verified against the repository on the assurance branch (not from documentation).
Line references are to the commit under review.

## Runtime dataset resolution (`mi_agent_api/data_source.py`)

Dataset location is entirely environment-driven; priority order in
`resolve_data_source()` (`data_source.py:159-182`):

1. `MI_AGENT_ANALYTICS_DATASET` (explicit MI-prepared CSV) — `prepared_explicit`
2. `MI_AGENT_PLATFORM_URI` → `MI_AGENT_PLATFORM_CANONICAL` → `MI_AGENT_PLATFORM_DIR`
   → `<repo>/out_platform/platform_canonical_typed.csv` — `platform_canonical`
3. `MI_AGENT_CENTRAL_TAPE`, or `MI_AGENT_ONBOARDING_OUTPUT_ROOT` +
   `MI_AGENT_CLIENT_ID` + `MI_AGENT_RUN_ID` — `central_tape`
4. `MI_AGENT_DATA_CSV` — `explicit_csv`
5. Fallback glob `synthetic_demo/**/*canonical_typed.csv`, `sorted()[0]` —
   `synthetic_demo`

Cache is signature-keyed (blob ETag or `path:mtime:size`), TTL
`MI_AGENT_DATA_CACHE_TTL` default 30s.

**Flags**

* The demo fallback picks whichever file sorts first; today that is the
  `multibook/platform_2026-05-31` (prior period) file, not the current
  2026-06-30 cut. Any file added under `synthetic_demo/` can silently change
  the default dataset. Production is protected by `trakt_core.policy`
  (synthetic refused in production mode), so this is a demo-correctness issue,
  not a production-tenancy issue — verified separately.
* Runtime mode default is `production` (fail closed); repo-root `conftest.py`
  sets `TRAKT_RUNTIME_MODE=test` for the suite; `trakt_core.runtime`
  refuses non-production mode when Azure markers are present.

## Tenant → dataset mapping

* `trakt_core/tenancy.py` is the authorisation gate; `config/tenancy.yaml`
  is optional and absent in the repo (only `config/tenancy.example.yaml`),
  so deployments run single-tenant with an open namespace scoped to the
  context tenant.
* Tenant id resolution for platform data: `mi_agent_api/datasets.py:111-122`
  (`MI_AGENT_CLIENT_ID` → parsed from `MI_AGENT_PLATFORM_URI` → `client_id`
  column → literal `platform`).
* `config/clients/` (plural) holds only `client_001/risk_limits_extracted.yaml`;
  `config/client/` (singular) holds master client configs — naming collision,
  no evidence of wrong resolution but worth keeping distinct.

## Synthetic datasets (fixture universes)

| Dataset | Rows | Reporting date(s) | Currency | Portfolios |
|---|---|---|---|---|
| `synthetic_demo/output/multibook/platform_2026-06-30_canonical_typed.csv` | 118 | 2026-06-30 | GBP | alp_acquired, alp_origination, spv1_sponsored |
| `synthetic_demo/output/multibook/platform_2026-05-31_canonical_typed.csv` | 116 | 2026-05-31 | GBP | same three |
| `synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv` | 36 | 2025-11-30 (file named 012026 — mismatch) | GBP | none (no provenance column) |
| `demo_platform/` (generated, not committed) | ~7k+ | 2026-04-30 / 05-31 / 06-30 | GBP | alp_origination, alp_acquired, SPV1 |
| `mi_agent/mi_query_harness.build_fixture(n=400, seed=20240611)` | 400 | none (no cut-off column) | none | none |

The in-memory 400-row harness fixture is the substrate for the existing
252-case calibration bank; it has no currency, reporting-date or provenance
columns, so those controls are untested by that bank.

## Existing question banks (pre-assurance)

* `config/mi/golden_questions/ere_mi_calibration_250.yaml` — 252 curated cases,
  enforced by `mi_agent/tests/test_mi_calibration_bank.py` (known_gap cases
  xfail with reason, never loosened).
* `config/mi/golden_questions/ere_mi_questions.yaml` — 350 generated cases,
  enforced by `mi_agent/tests/test_ere_golden_questions.py`; 17 categories;
  `arrears_default` and `nneg_er` are declared controlled-unsupported.
* `tests/fixtures/mi_interpreter/golden_questions.yaml` — 30 interpreter cases.
* `mi_agent/mi_query_harness.py` — registry-driven generated harness; core
  invariant: every recognised grouping dimension is APPLIED or REJECTED with
  reason, never silently dropped.

## Test inventory

~231 Python test files / ~3,870 test functions:
`tests/` 145 files (securitisation engine, onboarding, governance, phases),
`mi_agent_api/tests/` 50 files, `mi_agent/tests/` 30 files,
`tests/mi_agent_pptx/` 6 files. No `pytest.ini`/`pyproject.toml` — no declared
testpaths, markers or coverage gate.

**Flags**

* `mi_agent_api/tests/conftest.py` autouse fixture sets
  `MI_AGENT_AUTH_ENABLED=false` for the whole API suite; only `test_auth.py`
  re-enables it. Route-level authorisation is untested for the other 49 files.
* `landing-page/tests/demo_pack_reproducible_test.py` uses `*_test.py` suffix —
  not collected by default pytest discovery.

## Delivery surfaces

### React (`frontend/mi-agent-ui`)
* No client-side auth; relies on Azure Easy Auth injecting
  `X-MS-CLIENT-PRINCIPAL`. Tenant never sent by the client; portfolio always
  sent via a single `scoped()` query builder. `POST /mi/query` body:
  `{question, portfolio, portfolioId, asOfDate, datasetContext,
  sourcePortfolioLens, filters}`.
* Unset `VITE_AGENT_API_URL` → MockAgentClient demo mode (client-side only).
* Routes not exercised from React: `/mi/catalogue`, `/mi/pipeline/snapshots`,
  `/mi/pipeline/snapshot`, `/mi/evolution/compare`, `/mi/workspace/view`.

### Copilot (`deploy/copilot-agent` + `mi_agent_api/copilot_*`)
* Entra bearer token validated server-side (RS256/JWKS, issuer, audience,
  expiry); `TRAKT_COPILOT_AUTH_MODE=entra` default; unconfigured ⇒ 503 on all
  Copilot routes (fail closed).
* Only two plugin functions: `POST /v1/copilot/mi/query` (same in-process
  governed capability as React) and `GET /v1/copilot/artifacts/latest`.

### Artefacts / signed downloads
* `mi_agent_api/artefacts.py` `artefact.investor_pack.get` is the shared
  authorisation path for React deck download and Copilot artefacts: scope
  check, requested `client_id` only ever compared to `context.tenant_id`
  (TENANT_MISMATCH on conflict), `authorise_portfolio_access` before any path
  construction, tenant (never caller) selects the deck store.
* Signed download: HMAC-SHA256 token `{kind, client_id, expiry}`, TTL default
  300s clamped [60, 3600].

**Flags**

* `TRAKT_COPILOT_DOWNLOAD_SIGNING_KEY` unset ⇒ ephemeral per-process key
  (warning only) — breaks token redemption across multiple workers and on
  restart; deployment-checklist item.
* `GET /v1/copilot/artifacts/download` has no auth dependency; tenant comes
  from the signed token payload. Control = HMAC integrity + TTL. Requires an
  explicit assurance disposition.

### Operator console (`mi_agent_operator`)
* Separate FastAPI app; shared-secret token, fail-closed 503 when
  unconfigured; constant-time comparison; CORS only when explicitly
  configured. Not part of the client MI surface.

### PPTX generation (`mi_agent_pptx`)
* CLI/in-process only (no HTTP surface); client id derived from run artefact;
  authorisation happens downstream at `artefacts.py` when served.
