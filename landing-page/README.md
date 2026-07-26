# Trakt landing page

The public Trakt marketing site and its constrained synthetic demonstration.

A prospective client should understand Trakt in under two minutes: ask a
portfolio question, get an evidence-backed answer, see the metrics behind it,
request a report — then read what the full product does and why its output can be
relied on.

---

## Contents

- [Purpose](#purpose)
- [Architecture](#architecture)
- [Repository dependencies](#repository-dependencies)
- [Synthetic data sources](#synthetic-data-sources)
- [Trakt services reused](#trakt-services-reused)
- [Local setup](#local-setup)
- [Environment variables](#environment-variables)
- [Development commands](#development-commands)
- [Tests](#tests)
- [Production build](#production-build)
- [Deployment](#deployment)
- [Domain setup](#domain-setup)
- [Lead delivery setup](#lead-delivery-setup)
- [Analytics setup](#analytics-setup)
- [Demo safety controls](#demo-safety-controls)
- [Known limitations](#known-limitations)
- [Updating the synthetic portfolio values](#updating-the-synthetic-portfolio-values)
- [Replacing the demo video](#replacing-the-demo-video)
- [Enabling or disabling supported demo questions](#enabling-or-disabling-supported-demo-questions)

---

## Purpose

The page communicates three things, in this order:

1. **What a visitor experiences** — the interactive demonstration in
   `#live-demo`: ask, get a concise governed answer with KPIs / a chart / a
   table, request a management or investor report preview.
2. **What a buyer purchases** — a governed portfolio operating, analytics and
   reporting layer across eight capability areas, not a chatbot and not a
   dashboard.
3. **Why the output can be relied on** — deterministic calculation, governed
   source data, validation, provenance, auditability and controlled delivery.

The demonstration is **synthetic only**. It accepts no uploads, returns no
exposure-level records, and reaches no client environment.

---

## Architecture

```
                       BUILD TIME (developer machine / CI)
  ┌──────────────────────────────────────────────────────────────────────┐
  │ synthetic_demo/output/…canonical_typed.csv    (governed canonical)   │
  │                       ↓                                              │
  │ mi_agent_api.funded_prep.prepare_funded_mi_dataset                   │
  │                       ↓                                              │
  │ mi_agent.mi_agent_workflow.run_mi_agent_query   ← THE Trakt engine   │
  │                       ↓                                              │
  │ mi_agent_api.adapters.adapt_workflow_result                          │
  │                       ↓                                              │
  │ landing-page/scripts/build_demo_pack.py   (redact · cap · shrink)    │
  │                       ↓                                              │
  │ landing-page/data/demo-pack.json          (committed, ~60 kB)        │
  └──────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪═══════════ trust boundary ═══════════
                                    │
                       RUN TIME (public internet)
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Next.js App Router (server components + route handlers)              │
  │   /api/demo/meta    scope, suggestions, report actions, limits       │
  │   /api/demo/query   allow-listed intent match → pre-computed answer  │
  │   /api/demo/report  allow-listed report id    → preview pages        │
  │   /api/leads        validate → one configured delivery adapter       │
  │   /api/health       liveness                                         │
  └──────────────────────────────────────────────────────────────────────┘
```

The consequence worth stating plainly: **the Trakt MI engine is not deployed with
this site.** The public internet cannot reach `mi_service`, the FastAPI app,
Azure Blob Storage or any client environment, because none of them is part of
this deployment. Every figure on the page is nonetheless genuine Trakt output —
it was computed by the real deterministic engine when the pack was generated, and
a test proves the committed pack still matches a fresh build.

The landing page performs **no portfolio calculation of its own**, in TypeScript
or anywhere else. `src/lib/format.ts` formats values; it never derives them.

See [`docs/architecture.md`](docs/architecture.md) for component diagrams,
request flows, trust boundaries and the abuse-control design.

### Stack

Next.js 16 (App Router) · TypeScript · Tailwind CSS 4 · no charting library.

Tailwind 4 and TypeScript are already repository conventions
(`frontend/mi-agent-ui`). The design tokens in `src/app/globals.css` are the
existing Trakt palette (`#232D55` navy / `#919DD1` periwinkle, from
`analytics/charts_plotly.py`), carried over from
`frontend/mi-agent-ui/src/index.css` so the marketing site and the product read
as one system. Runtime dependencies are `next`, `react` and `react-dom` — nothing
else.

*Why not Vite + Static Web Apps, like `frontend/mi-agent-ui`?* This site needs
first-party server routes that hold secrets (lead delivery, rate-limit state,
session signing). SWA's static hosting cannot do that without a separate
Functions app. App Service is already an established target in this repository
(`trakt-mi-api`), so this adds no new Azure service type.

---

## Repository dependencies

The **runtime** app depends on nothing outside `landing-page/`. It ships
`data/demo-pack.json` and reads it in-process.

The **generator** (`scripts/build_demo_pack.py`) imports the Trakt engine and
therefore needs the repository root on `sys.path` (it adds it itself) and the
root `requirements.txt` installed. It is run by a developer or by CI, never by
the deployed site.

---

## Synthetic data sources

Everything below already existed in the repository; the landing page created
none of it.

| File | Used for |
|---|---|
| `synthetic_demo/config/config_client_SYNTHETIC_ERM.yaml` | client identity — **Synthetic Demo Lender**, originator ERE Funding Limited, UK equity release, GBP, reporting date 2025-11-30 |
| `synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv` | **the governed canonical dataset** every demo figure is computed from (36 exposures, £5,382,462.92) |
| `…_header_mapping_report.json` | Gate 1 evidence in the governance answer |
| `…_transform_report.json` | transform evidence (typed fields, parse failures) |
| `validation/…_field_summary.csv` | Gate 2/3 validation exceptions |
| `…_ESMA_Annex2_delivery_report.json` | Gate 5 delivery preflight (36 in / 36 out, 0 issues, PASS) |
| `frontend/mi-agent-ui/public/trakt-mark.svg` | the brand mark, inlined in `TraktWordmark.tsx` |
| `configs/pptx/investor_pack.yaml` | the real investor-pack slide order the report preview mirrors |

This is the same dataset `mi_agent_api/data_source.py` resolves as the bundled
demo (`KIND_SYNTHETIC_DEMO`) — the one the production Copilot action layer
*refuses* to answer from, which is exactly why it is the right dataset for a
public page.

---

## Trakt services reused

| Component | Module |
|---|---|
| Funded-dataset preparation (`ltv_bucket`, `age_bucket`, `ticket_bucket`, …) | `mi_agent_api.funded_prep.prepare_funded_mi_dataset` |
| Deterministic question parser | `mi_agent.llm_query_parser._deterministic_parse` (via the workflow) |
| MI query workflow — parse → validate → execute → chart | `mi_agent.mi_agent_workflow.run_mi_agent_query` |
| Query executor / validator / semantics registry | `mi_agent.mi_query_executor`, `mi_agent.mi_query_validator`, `mi_agent/mi_semantics_field_registry.yaml` |
| API response adapter (KPI/table/chart artifacts, display hints, reconciliation) | `mi_agent_api.adapters.adapt_workflow_result` |

This is the same chain that answers the React MI Agent and Microsoft 365 Copilot
(`docs/mi_shared_service_architecture.md`).

**Nothing in the existing Trakt services was modified.** The landing page is
additive: one new directory plus one new (manual-trigger) CI workflow.

---

## Local setup

```bash
cd landing-page
npm install
cp .env.example .env.local     # optional; the defaults run as-is
npm run dev                    # http://localhost:3000
```

The committed `data/demo-pack.json` means the site runs with **no Python
installed**. You only need the Trakt Python environment to regenerate it:

```bash
# from the repository root
pip install -r requirements.txt
python landing-page/scripts/build_demo_pack.py
```

---

## Environment variables

Full list with comments in [`.env.example`](.env.example). The ones that matter
in production:

| Variable | Required | Purpose |
|---|---|---|
| `NEXT_PUBLIC_SITE_URL` | **yes** | Canonical origin. Drives `metadataBase`, canonical link, Open Graph, `sitemap.xml`, `robots.txt`. No domain is hard-coded anywhere. Inlined at **build** time — set it for the build as well as the runtime. |
| `NEXT_PUBLIC_DEMO_NAME` | no | Display name of the synthetic client. |
| `APPLICATION_ENV` | **yes** | `production` enables secure cookies and makes the session secret mandatory. |
| `DEMO_SESSION_SECRET` | **yes in production** | HMAC key for the demo-session cookie. ≥32 chars. The app throws on boot in production without it rather than issuing forgeable sessions. |
| `DEMO_API_BASE_URL` | no | Reserved; unset today (the demo is served in-process). |
| `LEAD_DELIVERY_PROVIDER` | **yes** | `file` \| `email` \| `webhook` \| `console`. |
| `LEAD_NOTIFICATION_EMAIL`, `LEAD_FROM_EMAIL`, `EMAIL_API_KEY`, `EMAIL_API_URL` | for `email` | Transactional email delivery. |
| `CRM_WEBHOOK_URL`, `CRM_WEBHOOK_SECRET` | for `webhook` | CRM/automation delivery, optionally HMAC-signed. |
| `RATE_LIMIT_STORE_URL` | no | Shared rate-limit store for multi-instance deployments (see limitations). |
| `DEMO_MAX_QUESTIONS_PER_SESSION`, `DEMO_MAX_REPORTS_PER_SESSION`, `DEMO_SESSION_TTL_SECONDS`, `DEMO_MAX_QUESTION_LENGTH`, `DEMO_MAX_BODY_BYTES` | no | Demo session limits. |
| `RATE_LIMIT_DEMO_PER_MINUTE`, `RATE_LIMIT_REPORT_PER_MINUTE`, `RATE_LIMIT_LEAD_PER_HOUR` | no | Per-IP rate limits. |
| `NEXT_PUBLIC_ANALYTICS_PROVIDER`, `APPLICATIONINSIGHTS_CONNECTION_STRING` | no | Analytics adapter. Default `none`. |
| `NEXT_PUBLIC_DEMO_VIDEO_URL`, `NEXT_PUBLIC_DEMO_VIDEO_POSTER` | no | The overview video. Unset renders the placeholder. |

Only `NEXT_PUBLIC_*` values reach the browser. `src/lib/env.ts` is server-only
and must never be imported from a client component; `src/lib/public-config.ts`
exists so it never needs to be.

---

## Development commands

| Command | What it does |
|---|---|
| `npm run dev` | Development server on :3000 |
| `npm run build` | Production build (standalone output) |
| `npm start` | Serve the production build |
| `npm run lint` | ESLint (flat config, `next/core-web-vitals` + `next/typescript`) |
| `npm run typecheck` | `tsc --noEmit`, strict |
| `npm test` | Vitest unit + component suite |
| `npm run test:e2e` | Playwright, desktop + mobile, against a production build |
| `npm run demo-pack` | Regenerate `data/demo-pack.json` from the Trakt engine |
| `npm run demo-pack:check` | Fail if the committed pack is stale |

---

## Tests

**Unit and component** (Vitest + Testing Library) — `tests/`

- `intents.test.ts` — the allow-list matcher: supported phrasings reach the right
  intent; report phrasings reach the right action; unsupported topics are
  refused; and hostile input (SQL, template injection, path traversal, prompt
  injection) never reaches an answer.
- `api-demo.test.ts` — response schema for supported questions; controlled
  refusals; invalid portfolio id; oversized input and oversized body; malformed
  JSON; per-IP rate limiting; session question/report limits; session-cookie
  tampering; report-id traversal attempts; and pack-level assertions that no
  exposure-level column, internal query spec or file path is ever published.
- `api-leads.test.ts` — validation, consent, honeypot, minimum fill time,
  cross-origin rejection, rate limiting, oversized body.
- `components.test.tsx` — hero renders; demo suggestions load; a supported
  interaction displays the answer with metrics and provenance; an unsupported
  interaction displays a clear message; reset works; typed submission; server
  errors surface without breaking the demo; lead-form validation; the honeypot
  stays out of the accessibility tree; mobile navigation opens and closes.

**Demo-pack reproducibility** (pytest) —
`tests/demo_pack_reproducible_test.py`. Re-runs the generator and asserts the
committed pack is byte-identical, that the published totals equal the canonical
dataset to the penny, and that no exposure-level column or internal path leaks.
Skips automatically when the Trakt Python dependencies are absent.

**End-to-end** (Playwright) — `e2e/landing.spec.ts`, run on Desktop Chrome and
Pixel 7 against a real production build: page loads → visitor launches the demo →
selects a suggested question → answer and metrics appear → requests a report
preview and pages through it → reads the capability stack → submits the lead
form. Plus accessibility and layout checks (single `h1`, skip link focus, no
horizontal overflow) and the mobile menu.

```bash
npm test
npm run test:e2e
python -m pytest tests/demo_pack_reproducible_test.py -q   # from the repo root
```

In a sandbox without a downloaded browser, point Playwright at a pre-installed
one: `PLAYWRIGHT_CHROMIUM_PATH=/path/to/chromium npm run test:e2e`.

---

## Production build

```bash
NEXT_PUBLIC_SITE_URL=https://trakt.<your-domain> npm run build
npm start                       # honours $PORT, default 3000
```

`next.config.ts` sets `output: "standalone"`, so the build emits a
self-contained server in `.next/standalone` with only the modules it uses.

---

## Deployment

**Recommended: Azure App Service (Linux, Node 22).**

*Why.* The site needs server routes that hold secrets and keep rate-limit state,
which rules out Static Web Apps' static hosting without adding a separate
Functions app. App Service already runs `trakt-mi-api` in this repository, so
this introduces no new Azure service type, no new deployment idiom and no second
place to configure secrets. Container Apps is a supported alternative
(`Dockerfile`) if you would rather deploy an image; Static Web Apps is not
recommended for this app.

`.github/workflows/deploy-landing-page.yml` implements the App Service route:
lint → typecheck → unit tests → demo-pack reproducibility → build → assemble the
standalone bundle → deploy → smoke-check `/api/health`. Manual trigger by
default, matching `deploy-mi-api.yml`.

Set up once:

| What | Value |
|---|---|
| App Service | Linux plan, runtime stack **Node 22 LTS** |
| Startup command | `node server.js` (the assembled `package.json` sets `npm start` to this) |
| Repository secret | `AZURE_LANDING_PAGE_PUBLISH_PROFILE` (App Service → Get publish profile) |
| Repository variables | `AZURE_LANDING_PAGE_APP_NAME`, `LANDING_PAGE_SITE_URL` |
| App settings | the production variables above, plus `WEBSITES_PORT=8080` if you change the port |

**Health check path:** `/api/health` → `200 {"status":"ok", …}` when the demo
pack loaded. Configure it under App Service → Monitoring → Health check.

**Rollback:** App Service keeps previous deployments — *Deployment Center →
Deployment logs → Redeploy* on the last good commit. With deployment slots, swap
back to the previous slot. The site is stateless apart from in-memory
rate-limit counters, so rollback is immediate and loses nothing beyond those
counters.

**Container Apps alternative:**

```bash
docker build -t trakt-landing-page \
  --build-arg NEXT_PUBLIC_SITE_URL=https://trakt.<your-domain> landing-page/
# push to ACR, then create/update the Container App with:
#   target port 3000, ingress external, health probe /api/health,
#   min replicas 1 (see the rate-limit note under Known limitations)
```

> No live infrastructure was created or changed by this work. Everything above
> is configuration for you to apply.

---

## Domain setup

The build reads its origin from `NEXT_PUBLIC_SITE_URL`; **no domain is
hard-coded**, so any of `trakt.<your-domain>`, `www.<your-domain>` or
`demo.<your-domain>` works with no code change.

1. **DNS — ownership.** At your DNS provider add the TXT record App Service
   shows you (Custom domains → Add custom domain):
   `asuid.<subdomain>` → `<the verification id>`.
2. **DNS — routing.** For a subdomain (`trakt.` / `demo.`), add
   `CNAME <subdomain> → <app-name>.azurewebsites.net`.
   For an apex domain, use an `A` record to the App Service inbound IP plus the
   `asuid` TXT record — CNAME is not valid at the apex.
3. **Add the custom domain** in App Service → Custom domains → Add.
4. **TLS.** Create a free App Service Managed Certificate for the domain
   (Custom domains → Add binding → Create App Service Managed Certificate), bind
   it with SNI SSL, then set **HTTPS Only = On** and **Minimum TLS = 1.2**.
   Managed certificates renew automatically; they do not cover apex domains on
   every plan tier, so use an imported or Key Vault certificate for an apex.
5. **Set `NEXT_PUBLIC_SITE_URL` to the final origin and rebuild.** It is inlined
   at build time — changing only the App Service setting will leave the old
   origin in the canonical link, Open Graph tags and sitemap.
6. **If you publish under more than one hostname,** pick one canonical origin and
   redirect the others to it, so search engines see a single URL. The page emits
   a canonical link for `NEXT_PUBLIC_SITE_URL` regardless.

Propagation is usually minutes; allow up to 48 hours. Verify with
`curl -sI https://trakt.<your-domain>/api/health`.

---

## Lead delivery setup

`LEAD_DELIVERY_PROVIDER` selects exactly one adapter. **A submission is never
silently discarded** — an adapter either delivers or throws, and the route
reports a real failure to the visitor instead of a false success.

| Provider | Behaviour | Use |
|---|---|---|
| `file` (default) | Appends JSON Lines to `LEAD_STORE_DIR/leads.jsonl` | Development, and the documented development-safe adapter when no provider credentials exist |
| `email` | POSTs to a Resend-compatible JSON API (`EMAIL_API_URL`), `reply_to` set to the enquirer | **Preferred for production** |
| `webhook` | POSTs `{type, lead}` to `CRM_WEBHOOK_URL`, optionally signed `x-trakt-signature` (HMAC-SHA256 over the body) | CRM / automation platform |
| `console` | Structured log line | Test suites |

To enable email delivery:

```bash
LEAD_DELIVERY_PROVIDER=email
EMAIL_API_KEY=<provider key>          # App Service setting / Key Vault reference
LEAD_FROM_EMAIL=trakt-website@<your-domain>
LEAD_NOTIFICATION_EMAIL=sales@<your-domain>
```

Verify the sending domain with your provider (SPF/DKIM) or messages will be
rejected. `EMAIL_API_KEY` is read server-side only and never reaches the browser.

Microsoft Graph mail is deliberately **not** wired up: the repository's only
Entra integration (`mi_agent_api/copilot_auth.py`) is bearer-token *validation*
for the Copilot routes, not an application identity with `Mail.Send`. Adding one
is a tenant decision, so the preferred route here is the transactional provider.

`.leads/` is git-ignored. Do not commit captured leads.

---

## Analytics setup

`src/lib/analytics.ts` is an abstraction over three adapters, chosen by
`NEXT_PUBLIC_ANALYTICS_PROVIDER`:

| Value | Behaviour |
|---|---|
| `none` (default) | No-op. Records nothing, anywhere. No third-party tracker is loaded by default. |
| `appinsights` | Calls `window.appInsights.trackEvent` **if an operator has loaded the App Insights snippet**. This module never injects a script tag, so enabling it is an explicit deployment decision. |
| `firstparty` | `sendBeacon` to a same-origin `/api/analytics` collector. **The route is not implemented** — add it before selecting this value. |

**Events:** `hero_demo_click`, `demo_video_play`, `suggested_question_click`,
`typed_question_submit`, `answer_supported`, `answer_unsupported`,
`report_preview_request`, `session_limit_reached`, `demo_reset`,
`capability_card_open`, `book_demo_click`, `lead_form_submit`,
`lead_form_success`, `lead_form_error`.

**Data captured.** Only the event name plus a fixed, low-cardinality property
vocabulary: `intentId`, `reportId`, `capabilityId`, `section`, `source`,
`outcome`. Explicitly **not** captured: the text a visitor types into the demo
(intent identifiers are recorded instead), lead-form contents, any identifier
that persists across sessions, and any third-party cookie.

Server-side logging (`src/lib/http.ts` → `logRequest`) records route, outcome,
resolved intent id, the ephemeral session id and duration — **never the question
text**.

---

## Demo safety controls

| Control | Where |
|---|---|
| Synthetic data only; no upload path exists | by construction — there is no upload endpoint |
| Fixed portfolio allow-list (one portfolio) | `ALLOWED_PORTFOLIO_IDS`, `src/lib/demo-pack.ts` |
| Supported-intent allow-list; visitor text selects, never constructs, a query | `src/lib/intents.ts` |
| Report-id allow-list; no path, filename or parameter is accepted | `src/app/api/demo/report/route.ts` |
| No document download, no signed storage URL, no storage path | report previews are rendered in-page |
| Per-IP rate limits (demo 30/min, reports 10/min, leads 5/hour) | `src/lib/rate-limit.ts` |
| Session limits (12 questions, 3 report previews, 2-hour expiry) with a courteous message and a booking CTA | `src/lib/session.ts` |
| Max question length (240) and hard body ceiling (4 kB) enforced before parsing | `src/lib/http.ts` |
| Input sanitisation (control characters stripped, whitespace collapsed, length capped) | `src/lib/http.ts`, `src/lib/lead-validation.ts` |
| Signed HttpOnly session cookie; tampering is detected and the counters discarded | `src/lib/session.ts` |
| Generic error messages; no stack trace ever reaches a response | `src/lib/http.ts` |
| Secure headers — CSP, HSTS, `X-Frame-Options: DENY`, `nosniff`, Referrer-Policy, Permissions-Policy | `next.config.ts` |
| Same-origin only; no CORS permit is issued, and the lead route checks Origin/Referer | `next.config.ts`, `src/lib/http.ts` |
| Honeypot field + minimum form-fill time on the lead form | `src/lib/lead-validation.ts` |
| No client-side secrets — only `NEXT_PUBLIC_*` reaches the browser | `src/lib/public-config.ts` |
| No debug mode; `poweredByHeader: false`, no production source maps | `next.config.ts` |

The demo pack itself is redacted at generation time: no exposure-level
identifiers, no postcodes, no borrower ages, no internal query spec, no engine
identifiers, no file paths, and every row set capped at 12. Tests assert all of
it.

---

## Known limitations

- **Rate-limit state is in-process.** Correct for the documented single-instance
  App Service deployment. Scale out and each instance keeps its own counters,
  effectively multiplying every limit by the instance count.
  `RATE_LIMIT_STORE_URL` is read and reported by `src/lib/rate-limit.ts` but no
  shared-store backend is implemented — wire one before scaling out.
- **The `firstparty` analytics adapter has no collector route.** Do not select it
  until `/api/analytics` exists.
- **No demo video asset exists in the repository.** The section renders a
  documented placeholder; see below.
- **The synthetic portfolio is a single snapshot** of 36 exposures. Month-on-month
  movement, pipeline, funnel, forecast and arrears questions are therefore
  answered as controlled refusals rather than fabricated — deliberately, and the
  page shows this off.
- **Two engine dimensions are deliberately not offered**: interest-rate type and
  property type resolve to raw ESMA enumeration codes (`FXRL`, `RHOS`) with no
  display mapping in the repository, and origination vintage collapses to a
  single meaningless bucket on this dataset. Rather than relabel them here — which
  would be the landing page inventing semantics — they are omitted.
- **`npm audit` reports advisories** in the dev toolchain (`brace-expansion` via
  the ESLint plugin graph, `postcss` via Tailwind's build). Both are build-time
  only and not reachable at runtime; they clear when the upstream packages
  publish fixes.
- **Lead delivery has no retry or dead-letter queue.** A provider outage returns
  502 to the visitor with an alternative way to reach you; it does not queue.

---

## Updating the synthetic portfolio values

Every figure comes from the Trakt engine, so you never edit numbers by hand.

1. Change the source data or its pipeline output under `synthetic_demo/`.
2. Regenerate:
   ```bash
   pip install -r requirements.txt          # from the repository root
   python landing-page/scripts/build_demo_pack.py
   ```
3. Verify and commit:
   ```bash
   python -m pytest landing-page/tests/demo_pack_reproducible_test.py -q
   cd landing-page && npm test
   git add landing-page/data/demo-pack.json
   ```

The generator **fails loudly** if the engine refuses a question rather than
publishing a blank — a broken intent stops the build instead of shipping an
empty answer.

If you change the pack's *shape*, bump `PACK_VERSION` in the generator and
`EXPECTED_PACK_VERSION` in `src/lib/demo-pack.ts`. A mismatch throws at boot, so
a stale pack can never be served silently.

---

## Replacing the demo video

No `.mp4`/`.webm`/`.mov` and no hosted video URL exists anywhere in this
repository, so `src/components/site/DemoVideo.tsx` ships as a documented
placeholder: it renders a preview card that points at the live demonstration,
and the page is complete without it.

To add the final asset:

1. **Host it.** Either drop the file at
   `landing-page/public/media/trakt-overview.mp4` (fine up to ~20 MB; it is
   served by the app) or host it on a CDN / Azure Storage static website on a
   domain you control. Do **not** reference a local development path.
2. **Add a poster frame** at `landing-page/public/media/trakt-overview-poster.jpg`
   — with `preload="none"` the poster is all that loads until the visitor
   presses play, so it must exist to avoid an empty frame.
3. **Point at them:**
   ```bash
   NEXT_PUBLIC_DEMO_VIDEO_URL=/media/trakt-overview.mp4
   NEXT_PUBLIC_DEMO_VIDEO_POSTER=/media/trakt-overview-poster.jpg
   ```
   Both are inlined at build time — rebuild after setting them.
4. **If the asset is on another origin,** add that origin to `media-src` (and the
   poster's to `img-src`) in the CSP in `next.config.ts`. The CSP is `'self'`-only
   today and will otherwise block it.

The player never autoplays, always shows native controls, and falls back to the
placeholder card automatically if the source fails to load — so a broken or
missing asset degrades gracefully rather than leaving a dead frame.

---

## Enabling or disabling supported demo questions

The allow-list lives in one place: `INTENTS` in
`landing-page/scripts/build_demo_pack.py`.

**To add a question**, append an entry:

```python
{
    "id": "purpose_mix",
    "artifactTitle": "Current balance by purpose",
    "label": "Show the portfolio by loan purpose.",
    "category": "stratification",
    "question": "Show balance by purpose",       # sent to the real engine
    "phrases": ["by loan purpose", "purpose mix", "what were the loans for"],
    "narrative": "{top_label} is the largest at {top_value} ({top_pct}).",
    "followUps": ["funded_balance", "region_exposure"],
},
```

Then `python landing-page/scripts/build_demo_pack.py`. If the engine cannot
answer `question` against the synthetic dataset, the generator fails — which is
the point: a question only becomes public once Trakt can actually answer it.

Available narrative placeholders: `{kpi0}` (the KPI named by `primaryKpi`, else
the last), `{as_of}`, `{balance}`, `{loans}`, `{top_label}`, `{top_value}`,
`{top_pct}`, `{second_label}`, `{second_value}`.

**To remove a question**, delete its entry and any `followUps` reference to it —
a unit test fails if a follow-up names something that no longer exists.

**To change which questions appear as chips** without changing what is
answerable, edit `SUGGESTED_ORDER` in `src/lib/demo-pack.ts`. Questions not in
that list stay reachable by typing.

**To add a controlled refusal**, append to `CONTROLLED_UNSUPPORTED` with a
`reason` (why this demonstration will not answer) and a `productionNote` (what
Trakt does in a client environment). Refusals win ties against supported intents
in the matcher, so a visitor asking about arrears is told there are none rather
than handed an adjacent number.

---

## Directory layout

```
landing-page/
├── data/demo-pack.json           generated, committed — every published figure
├── docs/
│   ├── architecture.md           components, flows, trust boundaries, topology
│   ├── content-map.md            every visible claim → its repository evidence
│   └── implementation-note.md    the pre-build repository reconnaissance
├── e2e/landing.spec.ts           Playwright, desktop + mobile
├── scripts/build_demo_pack.py    the generator — runs the real Trakt engine
├── src/
│   ├── app/                      routes, layout, metadata, API handlers
│   ├── components/demo/          the interactive demonstration
│   ├── components/site/          nav, hero, video, content, lead form, footer
│   ├── lib/                      demo pack, intents, session, rate limits, leads
│   └── types/demo.ts             the public demo contract
└── tests/                        Vitest suites + the pytest reproducibility check
```
