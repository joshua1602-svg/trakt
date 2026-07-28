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
- [Rate-limit store setup](#rate-limit-store-setup)
- [Analytics setup](#analytics-setup)
- [Production configuration validation](#production-configuration-validation)
- [Liveness versus readiness](#liveness-versus-readiness)
- [Demo safety controls](#demo-safety-controls)
- [Known limitations](#known-limitations)
- [Updating the synthetic portfolio values](#updating-the-synthetic-portfolio-values)
- [Adding a product overview video](#adding-a-product-overview-video)
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
  │   /api/analytics    allow-listed events only                         │
  │   /api/health       liveness                                         │
  │   /api/ready        readiness (configuration gate)                   │
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

### The source is pinned and fails closed

`DEMO_SOURCE` in `scripts/build_demo_pack.py` names the dataset explicitly and
asserts its identity before a single figure is published: client id, client
name, portfolio id, asset class, currency, reporting date, a balance range, a
minimum exposure count, and a **SHA-256 of the canonical file**. Any mismatch
aborts the build with `Landing-page demo source mismatch`, naming what it
expected and what it found.

There is no fallback. `mi_agent_api.data_source` is not imported, so the
generator cannot silently resolve some other dataset; a test parses the module
and asserts it never will.

`src/lib/config.ts` enforces the same identity independently at runtime
(`EXPECTED_DEMO_SOURCE`), validating the pack it is about to serve — so a pack
built from the wrong portfolio cannot be served even if it were committed.

**To point the landing page at a different governed portfolio**, edit
`DEMO_SOURCE` and `EXPECTED_DEMO_SOURCE` together, then regenerate. Both are
deliberate, reviewable edits.

> **On the "£1.9bn demo-video portfolio".** A report during hardening said the
> page should use a ~£1.9bn portfolio from the Trakt demo video. An exhaustive
> search — every CSV with a balance column totalled programmatically, every
> fixture, every mock, all spreadsheets, git history, all branches, untracked
> and ignored files — found **no such dataset and no demo video anywhere in the
> repository**. The largest figures elsewhere (£842.6MM, £0.97BN in
> `frontend/mi-agent-ui/src/data/mockResponses.ts`) are hard-coded prose and
> literal arrays flagged `mock: true`, with no rows behind them. See
> `docs/implementation-note.md` § "Addendum — demo source provenance" for the
> full evidence table. Rather than fabricate a replacement, the selection was
> made explicit and fail-closed, as above.

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
| `RATE_LIMIT_STORE_URL` | **in production** | Shared rate-limit counter. Required unless `ALLOW_IN_MEMORY_RATE_LIMIT=true`. |
| `ALLOW_IN_MEMORY_RATE_LIMIT` | no | Deliberate acceptance of single-instance operation. Default `false`. |
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
| `npm run validate-config` | Exercise the production configuration rules |
| `npm run scan` | Scan the build output for secrets and internal references (run after `npm run build`) |

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
- `config.test.ts` — the production configuration rules: canonical-origin
  validation (https, bare origin, no loopback, no hard-coded Trakt hostname),
  session-secret strength, refusal of the `file`/`console` lead adapters,
  refusal of in-memory rate limiting without the override, App Insights without
  a connection string, and that error messages name variables but never values.
- `api-analytics.test.ts` — the event allow-list; that raw question text,
  answers and email addresses cannot be recorded even when sent; that no IP or
  user agent is recorded; oversized bodies, cross-origin beacons and malformed
  JSON; and campaign-attribution sanitisation.
- `lead-delivery.test.ts` — production refusal of development adapters; HTML and
  plain-text rendering with escaping; one bounded retry on a transient failure
  and none on a permanent one; that a provider failure throws rather than
  reporting success; idempotent resubmission; webhook signing; and that no
  credential appears in a thrown message.
- `api-ready.test.ts` — readiness healthy and degraded; that liveness stays 200
  while readiness returns 503; and that neither endpoint leaks paths, addresses,
  provider names or internal module names.
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

### Recommended hostname layout

| Host | Use |
|---|---|
| the main Trakt domain, or `www.` | **the landing page** — one canonical origin, with every alternate 301-redirecting to it |
| a separate hostname, e.g. `app.` | future authenticated client environments |

Keeping the public marketing site and authenticated client environments on
separate hostnames matters beyond tidiness: it keeps cookie scope, CSP and
session boundaries genuinely separate, so nothing served to the public internet
shares an origin with a client's data. `demo.` is available if you would rather
the demonstration sat apart from the corporate site — the code does not care,
because the origin is environment-driven throughout.

Propagation is usually minutes; allow up to 48 hours. Verify with
`curl -sI https://trakt.<your-domain>/api/health`.

---

## Lead delivery setup

`LEAD_DELIVERY_PROVIDER` selects exactly one adapter. **A submission is never
silently discarded** — an adapter either delivers or throws, and the route
reports a real failure to the visitor rather than a false success.

| Provider | Behaviour | Use |
|---|---|---|
| `email` | POSTs to a Resend-compatible JSON API, HTML + plain text, `reply_to` set to the enquirer | **Production (preferred)** |
| `webhook` | POSTs `{type, lead}` to `CRM_WEBHOOK_URL`, optionally signed `x-trakt-signature` (HMAC-SHA256 over the body), with an idempotency-key header | Production (CRM / automation) |
| `file` | Appends JSON Lines to `LEAD_STORE_DIR/leads.jsonl` | **Development only** |
| `console` | Structured log line | **Development only** (test suites) |

**`file` and `console` are refused in production.** Configuration validation
fails at startup, and `deliverLead` throws `CONFIG_LEAD_PROVIDER_UNSAFE` as a
last line of defence. A landing page that silently discards enquiries is worse
than one that will not start.

To enable email delivery:

```bash
LEAD_DELIVERY_PROVIDER=email
EMAIL_API_KEY=<provider key>          # App Service setting / Key Vault reference
LEAD_FROM_EMAIL=trakt-website@<your-domain>
LEAD_NOTIFICATION_EMAIL=sales@<your-domain>
```

Verify the sending domain with your provider (SPF/DKIM) or messages will be
rejected. `EMAIL_API_KEY` is read server-side only, travels in the
`Authorization` header (never the body), and never appears in a log line or an
error message — asserted by tests.

**What the adapter does**

* **Timeout** — `LEAD_DELIVERY_TIMEOUT_MS` (default 8 s), enforced by
  `AbortController`.
* **One bounded retry** on a transient failure (5xx, 408, 429, network), with a
  300 ms pause. A 4xx is not retried: it will not succeed. Deliberately one
  retry and not more — the visitor is waiting, and a provider that fails twice
  inside the timeout is down, not busy.
* **Unique submission id** (`randomUUID`) returned to the visitor as
  `reference`, and carried to the provider as an idempotency header, so a
  follow-up can be matched to a notification.
* **Idempotency** — an identical submission (same email, company, role and
  message) inside a 10-minute window is suppressed rather than delivered twice.
  Recorded only *after* a confirmed delivery, so a failure can still be retried.
  In-process and therefore per-instance: the cost of a miss is one duplicate
  notification, which is not worth a shared store and a new failure path on the
  submit route.
* **Structured internal logging, generic public response** — the provider's
  status is logged for an operator; the visitor sees a 502 with an alternative
  way to reach you.
* **Attribution** is attached server-side and never echoed back.

`.leads/` is git-ignored. Do not commit captured leads.

Microsoft Graph mail is deliberately **not** wired up: the repository's only
Entra integration (`mi_agent_api/copilot_auth.py`) is bearer-token *validation*
for the Copilot routes, not an application identity with `Mail.Send`. Adding one
is a tenant decision, so the preferred route here is the transactional provider.

### Who owns incoming leads

Enquiries go to exactly one destination — `LEAD_NOTIFICATION_EMAIL` or
`CRM_WEBHOOK_URL`. **Whoever owns that mailbox owns the response.** The form
tells the visitor "we use your details only to respond, and do not add you to a
marketing list"; honouring that is a business undertaking, not something the
code can enforce. If lead handling changes, that copy must change with it (it
is tracked in `docs/content-map.md`).

**Retention.** The app itself retains nothing: no lead is written to disk in
production, and the idempotency record holds a hash for ten minutes. Retention
is entirely a property of the destination you configure — set it there, and
document it in your privacy notice.

---

## Rate-limit store setup

Fixed-window limiting over a pluggable store (`src/lib/rate-limit.ts`).

| Adapter | When |
|---|---|
| in-memory | Development, and single-instance production with an explicit override |
| shared | Any deployment with more than one replica |

The shared adapter speaks a deliberately minimal HTTP contract, so it needs no
client library and adds no dependency:

```
POST {RATE_LIMIT_STORE_URL}
  { "key": "<opaque>", "windowSeconds": 60 }
→ 200
  { "count": 3, "resetAt": 1730000000000 }
```

Any Redis-backed counter, Azure Function or container satisfies it. The store is
never trusted for anything but counting: it receives an opaque key and returns
two numbers.

### Behaviour when the shared store is unavailable

A decision, stated rather than left implicit:

* **Lead submission fails closed.** A lead is a state change with a real cost to
  get wrong, and it is already the tightest limit on the site. If we cannot
  count, we refuse.
* **Demo queries degrade to the stricter local limit.** Marketing pages are
  read-mostly and the demo is served from a static pack; making the page
  unusable because a counter service blinked would trade a small abuse risk for
  a total loss of function. Each instance then enforces the full per-IP limit
  locally, so the worst case is N× the intended ceiling for the duration of the
  outage — bounded, and strictly better than open.
* **Analytics never fails the request**, under any condition.

### Single-instance operation

If no shared store is available, you may run **one** App Service instance with:

```bash
ALLOW_IN_MEMORY_RATE_LIMIT=true
```

This is a conscious, temporary arrangement and must be treated as such:

* it is correct **only** at one instance — set *Scale out* to a fixed count of 1
  and do not enable autoscale;
* with N replicas every limit is effectively multiplied by N, silently;
* counters reset on restart and on deployment;
* without the override, production **refuses to start** (`CONFIG_RATE_LIMIT_STORE_REQUIRED`),
  which is the point: scaling out is then a deliberate decision that forces you
  to provision a store first.

---

## Analytics setup

`src/lib/analytics.ts` is an abstraction over three adapters, chosen by
`NEXT_PUBLIC_ANALYTICS_PROVIDER`:

| Value | Behaviour |
|---|---|
| `none` (default) | No-op. Records nothing, anywhere. No third-party tracker is loaded. |
| `appinsights` | Uses `window.appInsights.trackEvent` if an operator loaded the snippet; otherwise falls back to the first-party collector so events are not silently lost. This module never injects a script tag. |
| `firstparty` | `sendBeacon` (with a `fetch` fallback) to this origin's `POST /api/analytics`. |

**The ten events**, and nothing else: `hero_demo_click`, `demo_open`,
`suggested_question_click`, `typed_question_submit`, `demo_answer_returned`,
`demo_refusal_returned`, `report_preview_opened`, `book_demo_click`,
`lead_submit_success`, `lead_submit_failure`.

**Data captured.** The event name plus a fixed, low-cardinality property
vocabulary: `intentId`, `reportId`, `refusalCategory`, `section`, `source`,
`outcome`. Each is stripped to `[A-Za-z0-9_.-]` and capped
at 64 characters — so a question string, an answer or an email address has no
field to travel in, by construction rather than by policy.

**Never captured:** raw question text (the intent id is recorded instead),
answer content, names, email addresses, IP addresses, user agents, any
identifier that outlives the session, any third-party cookie.

**The collector** (`/api/analytics`) is an allow-list, not an event sink:

* an unrecognised event name is dropped, so it cannot be used as
  internet-writable logging;
* same-origin only; 1 kB body ceiling; per-IP rate limit;
* **it always returns 204** — a rejected event, a rate-limited caller and a
  malformed body are indistinguishable. The browser has nothing useful to do
  with a failure, and an error would only tell an abuser what got through;
* analytics failure can never break the visitor experience.

Server-side request logging (`src/lib/http.ts` → `logRequest`) records route,
outcome, resolved intent id, the ephemeral session id and duration — never the
question text.

### Campaign attribution

`utm_source`, `utm_medium`, `utm_campaign`, `utm_content`, `utm_term`, plus the
first-party `persona`, `use_case` and `source`.

* Captured on arrival by `AttributionCapture`, mounted at the top of the page —
  not in the lead form, so a visitor who never scrolls that far is still
  attributed correctly.
* Held in `sessionStorage`, **for the current browsing session only**. Not a
  cookie and not `localStorage`: attribution should last exactly as long as the
  visit that carried it, and must not become a cross-site identifier.
* First write wins, so a mid-visit navigation cannot overwrite the campaign the
  visitor actually arrived on.
* Each value is capped at 64 characters and restricted to an allow-list of
  characters real campaign codes use. Disallowed characters become a space, so
  a newline injection collapses to separated words rather than running them
  together.
* **Re-sanitised server-side** — the browser is not trusted for this.
* Attached to the lead notification and **never echoed back to the browser**
  after submission.

---

## Production configuration validation

One validator, `src/lib/config.ts`, run at startup by `src/instrumentation.ts`.
Every unsafe production setting is an **error**, not a warning: a page that
boots with file-based lead capture, a forgeable session cookie or a localhost
canonical URL is worse than one that refuses to boot, because the failure is
silent and the loss is invisible.

| Code | Fires when |
|---|---|
| `CONFIG_SITE_URL_INVALID` | `NEXT_PUBLIC_SITE_URL` is missing, not https, carries a path/query/fragment, or is a loopback host |
| `CONFIG_DEMO_PACK_MISSING` | the pack is absent or has no intents |
| `CONFIG_DEMO_PACK_VERSION` | the pack's schema version is not the one this build expects |
| `CONFIG_DEMO_SOURCE_MISMATCH` | the pack was built from a different portfolio than this build is pinned to |
| `CONFIG_SESSION_SECRET_WEAK` | `DEMO_SESSION_SECRET` is unset, under 32 characters, or too uniform |
| `CONFIG_LEAD_PROVIDER_UNSAFE` | the provider is `file` or `console` |
| `CONFIG_LEAD_PROVIDER_INCOMPLETE` | the chosen provider is missing its credentials |
| `CONFIG_RATE_LIMIT_STORE_REQUIRED` | no shared store and no explicit single-instance override |
| `CONFIG_ANALYTICS_INVALID` | an unknown adapter, or App Insights with no connection string |
| `CONFIG_ENV_INVALID` | `APPLICATION_ENV` is not development/test/staging/production |

Messages name the **variable** and what is wrong with it — never a value.
Outside production the same checks run but only warn, so local development is
never blocked by the absence of a mail provider.

```bash
npm run validate-config   # exercises the rules with placeholder values
```

---

## Liveness versus readiness

| Endpoint | Answers | Fails when |
|---|---|---|
| `GET /api/health` | "is this process alive, and did it load a pack" | the pack is unusable |
| `GET /api/ready` | "is this process correctly configured to serve" | any configuration error above |

They must be able to disagree, and that is the point: a misconfigured lead
provider is a readiness problem, not a liveness one — restarting the process
would not fix it, so health deliberately stays 200 while readiness returns 503.

**Use `/api/health` as the App Service health probe.** The platform probe
restarts an instance that fails it, and restarting never fixes a configuration
error — a red `/api/ready` would put the site into a restart loop instead of
leaving it up and visibly not-ready.

**Use `/api/ready` in deployment smoke tests**, which is exactly what the CI
workflow does: it fails the deployment when configuration is wrong, at the point
where a human can act on it.

Both return coarse component states only. No file path, email address, service
URL, provider name, credential, stack trace or internal engine name — asserted
by tests.

```json
{
  "status": "ready",
  "environment": "production",
  "components": {
    "demoPack": "ready",
    "siteUrl": "configured",
    "session": "configured",
    "leadDelivery": "configured",
    "rateLimit": "shared",
    "analytics": "disabled"
  },
  "issues": []
}
```

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

- **No ~£1.9bn demo-video portfolio exists in this repository**, and neither
  does the demo video. The page publishes the only governed portfolio dataset
  present (`SYNTHETIC_ERE_Portfolio_012026`, 36 exposures, £5,382,462.92). See
  the note under [Synthetic data sources](#synthetic-data-sources) and the full
  evidence table in `docs/implementation-note.md`.
- **The synthetic portfolio is a single snapshot.** Month-on-month movement,
  pipeline, funnel, forecast and arrears questions are answered as controlled
  refusals rather than fabricated — deliberately, and the page shows this off.
- **Rate-limit state is in-process unless `RATE_LIMIT_STORE_URL` is set.**
  Production refuses to start without either a store or the explicit
  single-instance override. See
  [Single-instance operation](#single-instance-operation).
- **Idempotency for lead submission is per-instance.** A duplicate could slip
  through across replicas; the cost is one duplicate notification.
- **The App Insights adapter uses the browser snippet if present, else the
  first-party collector.** It never injects a script tag, so loading the snippet
  is an explicit deployment decision.
- **Two engine dimensions are deliberately not offered**: interest-rate type and
  property type resolve to raw ESMA enumeration codes (`FXRL`, `RHOS`) with no
  display mapping in the repository, and origination vintage collapses to a
  single meaningless bucket on this dataset. Relabelling them here would be the
  landing page inventing semantics.
- **`npm audit` reports advisories** in the dev toolchain (`brace-expansion` via
  the ESLint plugin graph, `postcss` via Tailwind's build). Both are build-time
  only and not reachable at runtime.
- **Lead delivery has no dead-letter queue.** A provider outage returns 502 to
  the visitor with an alternative way to reach you; it retries once and does not
  queue.
- **There is no overview video.** The placeholder section was removed rather
  than left telling visitors the walkthrough "will appear here" — see
  [Adding a product overview video](#adding-a-product-overview-video). Ship a
  transcript alongside the asset when it is added.

---

## Open investigations

Failures that occurred once and could not be reproduced are recorded in
[`docs/open-investigations.md`](docs/open-investigations.md), with what was
ruled out and the point at which a recurrence stops being a flake. One entry is
open: OI-1, a demo pack reported STALE in CI that reproduces cleanly everywhere
else. If it fires again it is treated as a determinism defect in the engine,
because "the same question returns the same number" is the claim it contradicts.

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

## Adding a product overview video

No `.mp4`/`.webm`/`.mov` and no hosted video URL exists anywhere in this
repository. The page previously carried a placeholder card reading "The recorded
walkthrough will appear here"; it was removed, because a placeholder tells a
prospect the site is unfinished. There is no video section today, and the page is
complete without one.

To add it back once an asset exists:

1. **Host it.** Either drop the file at
   `landing-page/public/media/trakt-overview.mp4` (fine up to ~20 MB; it is
   served by the app) or host it on a CDN / Azure Storage static website on a
   domain you control. Do **not** reference a local development path.
2. **Add a poster frame** at `landing-page/public/media/trakt-overview-poster.jpg`
   — with `preload="none"` the poster is all that loads until the visitor
   presses play, so it must exist to avoid an empty frame.
3. **Add the component and the section.** A `<video>` with `controls`,
   `preload="none"`, `playsInline` and a poster, wrapped in `Card`, placed after
   the example section in `src/app/page.tsx`. Read the URL from
   `publicConfig` (`src/lib/public-config.ts`), which no longer carries
   `videoUrl` / `videoPoster` — add them back alongside the component.
4. **If the asset is on another origin,** add that origin to `media-src` (and the
   poster's to `img-src`) in the CSP in `next.config.ts`. The CSP is `'self'`-only
   today and will otherwise block it.
5. **Ship a transcript** alongside it, and re-add `video_play` to
   `ANALYTICS_EVENTS` in `src/lib/analytics-events.ts` if you want play tracked.

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
