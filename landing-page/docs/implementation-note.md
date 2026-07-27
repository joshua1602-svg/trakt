# Repository reconnaissance and implementation note

Written before any code, from a search of the Trakt repository. Every claim
below cites the file it came from.

## 1. The synthetic client used in the existing Trakt demo materials

| Attribute | Value | Source |
|---|---|---|
| Client id | `synthetic_demo` | `synthetic_demo/config/config_client_SYNTHETIC_ERM.yaml` → `client.client_id` |
| Display name | **Synthetic Demo Lender** | same file → `client.display_name` |
| Originator legal entity | ERE Funding Limited | same file → `defaults.originator_name` |
| Asset class | UK equity release mortgages | `portfolio.asset_class: equity_release` |
| Country / currency | GB / GBP | `portfolio.country`, `portfolio.base_currency` |
| Reporting date | 2025-11-30 | `portfolio.static_reporting_date`, and the `data_cut_off_date` column of the produced canonical |

This is the client the shipped MI API serves by default: `mi_agent_api/data_source.py`
resolves `synthetic_demo/**/*canonical_typed.csv` as the bundled demo dataset
(`KIND_SYNTHETIC_DEMO`), and the Copilot action layer refuses to answer from it in
production (`copilot_actions._BLOCKED_SOURCE_KINDS`) — which is exactly why it is
the correct and safe dataset for a public marketing demo.

## 2. The synthetic portfolio

`SYNTHETIC_ERE_Portfolio_012026` — 36 exposures, £5,382,462.92 current
outstanding balance as at 30 November 2025.

Source files (all already in the repository, none created by this project):

| File | Role |
|---|---|
| `synthetic_demo/input/SYNTHETIC_ERE_Portfolio_012026.csv` | raw synthetic loan tape (pipeline input) |
| `synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv` | **the governed canonical output** — the dataset every demo answer is computed from |
| `synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_header_mapping_report.json` | Gate 1 semantic-alignment evidence (raw header → canonical field, confidence) |
| `synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_transform_report.json` | transform evidence (per-field format, nulls, parse failures) |
| `synthetic_demo/output/validation/SYNTHETIC_ERE_Portfolio_012026_field_summary.csv` | Gate 2/3 validation exceptions |
| `synthetic_demo/output/validation/SYNTHETIC_ERE_Portfolio_012026_dashboard.json` | validation summary counts |
| `synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_ESMA_Annex2_delivery_report.json` | ESMA Annex 2 delivery preflight (36 rows in / 36 out, 0 issues, PASS) |

Dimensional variety actually present (checked, not assumed): region 10 values,
LTV band 4, borrower-age band 6, origination channel 3, ticket-size band 4,
interest-rate type 2, vintage year 2. Occupancy type and amortisation type are
single-valued and are therefore **not** offered as demo dimensions.

## 3. Existing MI / Copilot endpoints and actions

`mi_agent_api/app.py` (FastAPI, deployed as the `trakt-mi-api` Azure App Service)
plus `mi_agent_api/copilot_actions.py`, which exposes exactly three Microsoft 365
Copilot v1 actions (`docs/copilot_v1_implementation.md`):

| Action | Route | Contract |
|---|---|---|
| `askTraktMi` | `POST /v1/copilot/mi/query` | `{question, portfolioId?, asOfDate?}` → governed envelope (answer, interpreted, querySpec, reportingDate, validation, reconciliation, `supportingValues` = kpi/table/chart artifacts) |
| `getLatestInvestorDeck` | `GET /v1/copilot/artifacts/latest/investor-deck` | metadata + short-lived HMAC-signed download URL |
| `getLatestCanonicalTape` | `GET /v1/copilot/artifacts/latest/canonical-tape` | metadata + signed URL |

Both channels (React workspace and Copilot) are thin adapters over the one shared
governed service `mi_agent_api.mi_service.execute_governed_mi_query`
(`docs/mi_shared_service_architecture.md`).

**This API is not safe for direct public use.** It requires an Entra ID bearer
token and fails closed (503) without one (`mi_agent_api/copilot_auth.py`); the
Copilot MI action additionally refuses the synthetic dataset by design; and it is
a single-tenant deployment pointed at real client data in production. Per the
brief, the landing page therefore uses **a constrained public-demo adapter that
calls the existing deterministic services internally** — see section 6.

## 4. Chart / metric generation functions reused

| Component | File | Use here |
|---|---|---|
| Deterministic question parser | `mi_agent/llm_query_parser._deterministic_parse` | via the workflow below |
| MI query workflow (parse → validate → execute → chart) | `mi_agent/mi_agent_workflow.run_mi_agent_query` | **every demo answer** |
| Query executor / validator / semantics registry | `mi_agent/mi_query_executor.py`, `mi_agent/mi_query_validator.py`, `mi_agent/mi_semantics_field_registry.yaml` | invoked by the workflow |
| Chart factory | `mi_agent/mi_chart_factory.py` | invoked by the workflow |
| Funded-dataset preparation (derives `ltv_bucket`, `age_bucket`, `ticket_bucket`, `vintage_year`, …) | `mi_agent_api/funded_prep.prepare_funded_mi_dataset` | applied before every query, exactly as the API does |
| API response adapter (kpi / table / chart artifacts, display hints, reconciliation, provenance) | `mi_agent_api/adapters.adapt_workflow_result` | the envelope the demo pack is built from |
| Brand palette | `analytics/charts_plotly.py` → `#232D55` navy, `#919DD1` periwinkle; tokens in `frontend/mi-agent-ui/src/index.css` | the landing page's design tokens |

## 5. Report generation

`mi_agent_pptx/` builds the **Investor & Funder MI Pack**; the slide order is
declared in `configs/pptx/investor_pack.yaml` (cover → executive summary →
stratifications I–III → multi-dimensional risk → geography → funded evolution →
vintage cohorts → pipeline → funnel → forecast → risk limits → methodology →
appendix). The public demo mirrors **only the slides the single-snapshot
synthetic dataset can genuinely support** and renders them as in-page previews;
it never produces a downloadable document (see section 7).

## 6. Reuse strategy — how the demo stays deterministic without exposing the API

A build-time generator, `landing-page/scripts/build_demo_pack.py`, imports the
real Trakt engine in-process, runs each allow-listed question against the
bundled synthetic canonical, and writes the resulting governed envelopes to
`landing-page/data/demo-pack.json`. The runtime Node backend serves *those*
answers through an allow-listed intent matcher.

Consequences, all of them deliberate:

* every number on the page is produced by the same deterministic executor that
  answers the React workspace and Microsoft 365 Copilot — no re-implementation
  of portfolio maths in the landing page, in TypeScript or anywhere else;
* the public internet never reaches `mi_service`, the FastAPI app, Azure Blob
  Storage, or any client environment;
* answers are reproducible: re-running the generator on the same canonical must
  reproduce the same pack (asserted by a test).

## 7. Authentication, deployment and environment conventions found

* **Auth**: Easy-Auth `X-MS-CLIENT-PRINCIPAL` for `/mi/*` (`mi_agent_api/auth.py`);
  Entra bearer for `/v1/copilot/*` (`copilot_auth.py`). The landing page is
  deliberately **public and unauthenticated**, which is why every control in
  section 7 of the brief is implemented instead.
* **Deployment**: Azure Static Web Apps for the React workspace
  (`.github/workflows/azure-static-web-apps-*.yml`, `app_location: frontend/mi-agent-ui`),
  Azure App Service for the API (`.github/workflows/deploy-mi-api.yml`,
  `deploy/trakt-mi-api/`), Azure Container/Web App for Streamlit
  (`Dockerfile.streamlit`, `deploy-streamlit.sh`).
* **Env vars**: `MI_AGENT_*` (data source, auth, deck root), `TRAKT_COPILOT_*`
  (Copilot auth, download signing), `VITE_*` (client-visible frontend config),
  `AZURE_STORAGE_*` / `MI_AGENT_PLATFORM_URI` (blob).
* **Storage**: `processed-v2/{platform,decks}/{client}/latest/…` blob convention.

## 8. What does not exist in the repository

* **No demo video asset.** No `.mp4`/`.webm`/`.mov` anywhere, and no hosted video
  URL in any config. The page therefore ships a documented placeholder video
  component (`DemoVideo.tsx`) that degrades to a static interface preview, with
  the exact drop-in location documented in the README.
* **No second reporting period** for the synthetic portfolio, so month-on-month
  movement, pipeline, funnel and forecast questions cannot be answered honestly.
  They are wired as *controlled unsupported* responses rather than fabricated.
* **No brand font, logo lockup or screenshots** beyond
  `frontend/mi-agent-ui/public/trakt-mark.svg`; the wordmark is set in the
  existing UI's typeface stack (Inter) using the existing palette.

## 9. Chosen stack

Next.js 16 (App Router) + TypeScript + Tailwind CSS 4 + Recharts, deployed to
**Azure App Service (Linux, Node 22)**.

*Why not Vite/Static Web Apps like `frontend/mi-agent-ui`?* The landing page needs
first-party server routes that hold secrets (lead delivery, rate-limit state,
session signing). SWA's static hosting cannot do that without adding a separate
Functions app; App Service is already an established deployment target in this
repository (`trakt-mi-api`), so this adds no new Azure service type. Tailwind 4 and
TypeScript are already repository conventions
(`frontend/mi-agent-ui/package.json`); no charting library is used (see the
note in `src/components/demo/Artifacts.tsx`). A Dockerfile is also provided for Azure
Container Apps.

---

# Addendum — demo source provenance

Added during production hardening, in response to a report that the landing
page was using the wrong synthetic portfolio: that it should use the
**~£1.9bn portfolio from the Trakt demo video** rather than the £5.38m
`SYNTHETIC_ERE_Portfolio_012026`.

## The search

The repository was searched exhaustively for that portfolio and for the demo
video itself:

| Search | Result |
|---|---|
| `1.9bn`, `£1.9bn`, `1.9B`, `1900000000`, `1,900,000,000` and near variants | **no match** |
| any integer between 1.8x10^9 and 2.0x10^9, any file type | **no match** |
| every CSV in the repository with a balance column, totalled programmatically | 12 files; **largest is £5,382,462.92 / 36 exposures** |
| `*.xlsx`, `*.pptx`, `*.potx` | only the ESMA XSD template and the NUTS lookup — no portfolio |
| demo-video scripts, storyboards, transcripts, voiceover, screencasts | **none exist** |
| `.mp4` / `.webm` / `.mov`, or a hosted video URL in any config | **none exist** |
| Copilot test fixtures (`mi_agent_api/tests/test_copilot_actions.py`) | a 1-loan, £100,000 tape |
| declarative-agent package (`deploy/copilot-agent/*`) | no portfolio totals at all |
| git history for deleted datasets; all branches; untracked and ignored files | **nothing** |

## What the large figures elsewhere actually are

`frontend/mi-agent-ui/src/data/mockResponses.ts` contains £842.6MM, £84.2MM and
£0.97BN. These are **not** a portfolio:

* they are hard-coded strings inside narrative prose;
* the artifacts they accompany are literal arrays (`EXEC_KPIS`, `REGION_ROWS`)
  in `mockArtifacts.ts`;
* every artifact is stamped `mock: true`;
* they are served by `MockAgentClient`, the React workspace's offline mode.

There are no rows behind them. Using them would have meant copying displayed
totals with no traceable data source, and fabricating the distributions the
landing page publishes — both explicitly ruled out.

## Conclusion

**The ~£1.9bn video-demo portfolio does not exist in this repository, and
neither does the demo video.** `SYNTHETIC_ERE_Portfolio_012026` is not a
"generic small dataset" chosen by default — it is the *only* governed portfolio
dataset present, and it is the one `mi_agent_api/data_source.py` resolves as
the bundled demo.

Rather than fabricate a replacement, the selection was made **explicit and
fail-closed** instead. `DEMO_SOURCE` in `scripts/build_demo_pack.py` names the
dataset, its client, its portfolio id, its reporting date, its currency, its
asset class, its expected balance range, a minimum exposure count and a SHA-256
of the canonical file. Any mismatch aborts the build with
`Landing-page demo source mismatch`. There is no fallback path — `data_source`
is not imported, and a test asserts it never will be.

The practical consequence: **if the £1.9bn dataset is added to the repository,
edit `DEMO_SOURCE` and re-run the generator.** If someone points the generator
at the wrong dataset, it refuses. A test proves this by pinning the expected
range to 1.8-2.0bn and asserting the generator rejects the £5.38m portfolio:

```
BLOCKED  1.8bn-2.0bn expected -> total balance 5,382,462.92 outside the
                                 expected range 1,800,000,000-2,000,000,000
```

The runtime enforces the same identity independently: `EXPECTED_DEMO_SOURCE` in
`src/lib/config.ts` validates the pack it is about to serve, so a pack built
from the wrong portfolio cannot be served even if it were committed.
