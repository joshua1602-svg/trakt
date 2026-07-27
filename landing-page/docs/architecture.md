# Landing page — architecture

How the public Trakt site produces genuine Trakt figures without exposing any
Trakt service to the internet.

---

## 1. Component diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  BUILD TIME — developer workstation or CI. Never runs in production.          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv     ║
║  synthetic_demo/output/*_header_mapping_report.json                           ║
║  synthetic_demo/output/*_transform_report.json                                ║
║  synthetic_demo/output/validation/*_field_summary.csv                         ║
║  synthetic_demo/output/*_ESMA_Annex2_delivery_report.json                     ║
║                              │                                               ║
║                              ▼                                               ║
║   ┌────────────────────────────────────────────────────────────────────┐     ║
║   │ scripts/build_demo_pack.py                                         │     ║
║   │                                                                    │     ║
║   │   for each allow-listed intent:                                    │     ║
║   │     prepare_funded_mi_dataset(df)        ← mi_agent_api.funded_prep │     ║
║   │     run_mi_agent_query(question, df, …)  ← mi_agent (THE engine)    │     ║
║   │     adapt_workflow_result(workflow)      ← mi_agent_api.adapters    │     ║
║   │     redact · cap rows · drop spec/paths/diagnostics                 │     ║
║   └────────────────────────────────────────────────────────────────────┘     ║
║                              │                                               ║
║                              ▼                                               ║
║              data/demo-pack.json   (~60 kB, committed)                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
                               │
                               │  committed artefact — the ONLY thing that
                               │  crosses into the deployed application
                               ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  RUN TIME — Azure App Service (Linux, Node 22). Public internet.              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Browser                                                                    ║
║     │  server-rendered HTML (demo metadata already embedded)                 ║
║     │  fetch POST /api/demo/query · /api/demo/report · /api/leads            ║
║     ▼                                                                        ║
║   ┌────────────────────────────────────────────────────────────────────┐     ║
║   │ Next.js App Router                                                 │     ║
║   │                                                                    │     ║
║   │  app/page.tsx           server component → buildMeta()             │     ║
║   │  app/api/demo/meta      scope · suggestions · actions · limits     │     ║
║   │  app/api/demo/query     rate limit → sanitise → resolve → answer   │     ║
║   │  app/api/demo/report    rate limit → allow-list id → pages         │     ║
║   │  app/api/leads          origin → rate limit → validate → deliver   │     ║
║   │  app/api/analytics      allow-listed events, always 204            │     ║
║   │  app/api/health         liveness                                   │     ║
║   │  app/api/ready          readiness — the configuration gate         │     ║
║   │                                                                    │     ║
║   │  lib/demo-pack.ts   the pack + allow-lists  (server-only)          │     ║
║   │  lib/intents.ts     deterministic phrase matcher (server-only)     │     ║
║   │  lib/session.ts     signed cookie, question/report counters        │     ║
║   │  lib/rate-limit.ts  fixed-window over a pluggable store            │     ║
║   │  lib/leads.ts       one delivery adapter, retry + idempotency      │     ║
║   │  lib/config.ts      production configuration validator             │     ║
║   │  lib/attribution.ts campaign capture, sanitised server-side        │     ║
║   │  instrumentation.ts startup validation — fails closed              │     ║
║   └────────────────────────────────────────────────────────────────────┘     ║
║                              │                                               ║
║                              ▼                                               ║
║       lead delivery: email API │ CRM webhook │ local JSONL │ log             ║
╚══════════════════════════════════════════════════════════════════════════════╝

  NOT REACHABLE FROM THE DEPLOYED SITE — not merely blocked, absent:
    mi_agent_api (FastAPI) · mi_service · mi_agent · Azure Blob Storage ·
    processed-v2/{platform,decks}/… · any client environment · any Entra tenant
```

The Trakt engine is a **build-time dependency, not a runtime one**. The deployed
bundle contains no Python, no pandas, no Trakt module and no Azure SDK.

---

## 2. Request flows

### 2.1 First paint

```
GET /
  → app/layout.tsx  metadata · structured data · design tokens
  → app/page.tsx    (server component)
      buildMeta()   reads data/demo-pack.json in-process
      renders Nav, Hero (with this portfolio's real figures),
              content sections, CopilotDemo(meta), LeadForm, Footer
  ← HTML, complete
```

The demo metadata is embedded in the server-rendered payload, so the page paints
with the synthetic scope, the suggested questions and the report actions already
present. No client fetch on load, no skeleton, no layout shift.

### 2.2 A demo question

```
POST /api/demo/query  {questionId} | {question}
 1. clientIp(headers)                          x-forwarded-for, left-most
 2. checkRateLimit("query:<ip>", 30/min)       → 429 + Retry-After
 3. readJsonBody(max 4 kB)                     ceiling applied BEFORE parsing
                                               → 413 oversized · 400 malformed
 4. portfolioId ∈ ALLOWED_PORTFOLIO_IDS        → 400 "Unknown portfolio."
 5. length ≤ DEMO_MAX_QUESTION_LENGTH          → 400 (rejected, not truncated)
 6. sanitiseText                               control chars out, whitespace
                                               collapsed, length capped
 7. loadSession(cookie)                        signature checked; a bad
                                               signature yields a FRESH session,
                                               never trusted counters
 8. session.q ≥ limit                          → status "limit_reached" + CTA
 9. resolveId(id) | resolveQuestion(text)      ← the only "understanding" step
      ├─ intent      → pre-computed governed answer from the pack
      ├─ report      → the report action's pointer answer
      ├─ unsupported → controlled refusal + what production does instead
      └─ unmatched   → "will not guess" + suggestions
10. increment counters, re-sign the cookie
11. logRequest{route, outcome, intentId, sessionId, durationMs}
                                               ← never the question text
  ← 200 DemoAnswerResponse
```

**Step 9 is the whole security story.** `resolveQuestion` is a deterministic
scorer over a fixed phrase table (`lib/intents.ts`). Visitor text *selects* one
of a fixed set of pre-computed answers, or selects nothing. It is never
interpolated into a query, never sent to a model, never sent to the Trakt engine
— which is not deployed here. There is no code path from visitor input to
portfolio computation.

Refusals win ties against supported intents, because answering the adjacent
question is worse than declining: someone asking about arrears should be told
there are none, not handed a balance.

### 2.3 A report preview

```
POST /api/demo/report  {reportId}
 1. rate limit "report:<ip>", 10/min
 2. body ceiling
 3. sanitiseText(reportId, 64)
 4. reportId ∈ ALLOWED_REPORT_IDS               ← allow-list, before anything
                                                  else. No path is parsed, no
                                                  file is opened, no parameter
                                                  reaches a filesystem call.
 5. session.r ≥ limit → "limit_reached" + CTA
 6. return the pack's pre-rendered preview pages
  ← 200 DemoReportResponse   no URL · no signed link · no storage path · no file
```

`../../etc/passwd`, `/etc/passwd`, `investor_report; ls` and `..` all fail step 4
with the same generic `400 {"error":"Unknown report."}`. Tests assert each one.

### 2.4 A lead

```
POST /api/leads
 1. isSameOrigin(request)                       Origin, else Referer → 403
 2. rate limit "lead:<ip>", 5/hour
 3. readJsonBody(max 8 kB)
 4. botSignals: honeypot filled | elapsedMs < LEAD_MIN_FILL_MS
       → 202 {"status":"received"}              same shape a human sees, so a
                                                bot learns nothing
 5. validateLead                                → 400 with per-field errors
 6. deliverLead(lead)                           exactly one configured adapter
       success → 201 {"status":"received"}
       throw   → 502 + an alternative way to reach us; reason logged
                 server-side only
```

A submission is never silently discarded.

---

## 3. Trust boundaries

| # | Boundary | Crosses it | Enforced by |
|---|---|---|---|
| **A** | Public internet → the app | HTTP requests | CSP, HSTS, `X-Frame-Options: DENY`, `nosniff`, Referrer-Policy, Permissions-Policy, no CORS permit issued, same-origin check on the lead route |
| **B** | Visitor text → what the demo says | an allow-listed intent id, or nothing | `lib/intents.ts` — deterministic matcher over a fixed phrase table |
| **C** | Build time → run time | `data/demo-pack.json`, and nothing else | generated by `build_demo_pack.py` with redaction; asserted reproducible by pytest |
| **D** | The app → the Trakt platform | **nothing** | the Trakt engine, the MI API and Azure Storage are not dependencies of the deployed bundle |
| **E** | Server config → the browser | `NEXT_PUBLIC_*` only | `lib/env.ts` is `server-only`; `lib/public-config.ts` carries the client-safe subset |
| **F** | The app → lead destination | one validated lead | one adapter, server-side credentials, 8 s timeout, optional HMAC signature |

Boundary **D** is the one that matters most: it is enforced by absence rather
than by a rule, and absence cannot be misconfigured.

---

## 4. Public versus private

| Public (this deployment) | Private (unchanged, untouched) |
|---|---|
| Marketing content | The Trakt MI Agent workspace (`frontend/mi-agent-ui`) |
| 11 allow-listed synthetic answers | The full MI query surface — free-form questions, drill-through, filters, source-portfolio lenses |
| 2 report previews, rendered in-page | Investor pack / canonical tape generation and delivery (`mi_agent_pptx`, `mi_agent_api/decks.py`) |
| Aggregated measures only | Exposure-level records |
| Single snapshot, 36 synthetic exposures | Governed snapshot history, temporal comparison, cohorts, forecasts, risk limits |
| No authentication (nothing to protect) | Entra ID bearer tokens, Easy-Auth headers, per-client deployments |
| No storage | Azure Blob `processed-v2/{platform,decks}/{client}/…` |

The public demo is deliberately the smallest credible slice. It shows the Copilot
*action layer* — ask, answer, request a report — not the workspace, which is the
product's differentiation and stays behind the sale.

---

## 5. Data flow

**Into the pack (build time).** The generator reads the governed canonical
dataset, applies the production funded-dataset preparation, runs each
allow-listed question through the deterministic engine, and takes the adapter's
governed envelope. It then *removes*: the internal `MIQuerySpec`, engine
identifiers and labels, diagnostics, warnings, artifact ids, creation timestamps,
native chart types, source paths, and every column on the forbidden list
(loan/borrower/collateral identifiers, postcodes, borrower ages, the originator
LEI). Row sets are capped at 12. What survives is aggregated display values plus
the column format contract and the engine's own balance-coverage figure.

**Out of the app (run time).** Only what a `DemoAnswerResponse` or
`DemoReportResponse` carries: the written answer, the KPI/chart/table artifacts,
the as-of date, the portfolio scope, follow-up suggestions, a synthetic flag and
the session usage counters.

**Never in either direction.** Client data, exposure-level records, raw API
payloads, internal traces, prompts, infrastructure details, validation
configuration, admin functions, file uploads, report-download URLs, storage
credentials.

**Retained by the app.** Nothing about the visitor beyond a signed cookie holding
four values — an opaque session id, an issue time and two counters. No question
text is stored or logged. Leads are retained only by the configured delivery
destination.

---

## 6. Abuse-control design

Layered, so no single control is load-bearing:

1. **Structural** — there is no upload endpoint, no free-form query path, no file
   parameter and no engine to reach. Most attacks have no surface.
2. **Allow-lists** — one portfolio, eleven intents, two reports. Everything is
   checked by exact membership before any work happens.
3. **Per-IP rate limits** — demo 30/min, reports 10/min, leads 5/hour. Fixed
   window, `Retry-After` on 429. Missing proxy headers collapse callers into one
   bucket, which fails *stricter*, not open.
4. **Per-session limits** — 12 questions, 3 report previews, 2-hour expiry,
   carried in a signed HttpOnly cookie. A tampered cookie is discarded and a
   fresh session minted, so counters cannot be reset by editing it. On reaching a
   limit the visitor gets a courteous message and a booking CTA, not an error.
5. **Input bounds** — 240-character questions, 4 kB demo bodies, 8 kB lead
   bodies, enforced before parsing. Oversized questions are rejected, not
   truncated.
6. **Sanitisation** — control characters stripped, whitespace collapsed, length
   capped. Output escaping is React's; values are rendered as text, never as
   HTML.
7. **Bot controls on the lead form** — honeypot field (hidden from sighted users
   *and* from assistive technology) plus a minimum form-completion time, both
   enforced server-side, both answering `202 {"status":"received"}` so a bot
   cannot tell it failed.
8. **Response hygiene** — generic error messages, no stack trace, no internal
   identifier, `poweredByHeader: false`, no production source maps, `no-store`
   and `X-Robots-Tag: noindex` on every API response.
9. **Transport and browser** — strict CSP (`'self'` everywhere; no `unsafe-eval`
   in production; `frame-ancestors 'none'`), HSTS with preload, `nosniff`,
   `strict-origin-when-cross-origin`, a Permissions-Policy that denies camera,
   microphone, geolocation and topics.
10. **Observability** — structured request logs carrying route, outcome, intent
    id, session id and duration. Never the question text.

---

## 7. Deployment topology

```
                      DNS: trakt|www|demo.<domain>
                                  │  CNAME (subdomain) / A + asuid TXT (apex)
                                  ▼
                    ┌──────────────────────────────┐
                    │  Azure App Service           │
                    │  Linux · Node 22 LTS         │
                    │  HTTPS only · TLS ≥ 1.2      │
                    │  Managed certificate         │
                    │  Health check: /api/health   │
                    │                              │
                    │  node server.js              │
                    │   (Next standalone bundle    │
                    │    + data/demo-pack.json)    │
                    └──────────────────────────────┘
                                  │
                                  ▼  outbound only, one hop
                    email API │ CRM webhook │ local JSONL
```

**Recommended: App Service.** The site needs server routes holding secrets and
rate-limit state, which Static Web Apps' static hosting cannot provide without a
separate Functions app. App Service already runs `trakt-mi-api` in this
repository — no new service type, no new deployment idiom, no second place to
keep secrets. `Dockerfile` supports Azure Container Apps as an alternative.

**Instances.** Run one instance until a shared rate-limit store is wired
(`RATE_LIMIT_STORE_URL` is read but unimplemented); with N instances each keeps
its own counters and every limit is effectively multiplied by N.

**Statelessness.** Apart from those counters the app holds nothing, so rollback
is a redeploy or a slot swap and loses nothing of value.

**CI.** `.github/workflows/deploy-landing-page.yml`: lint → typecheck → unit
tests → demo-pack reproducibility → build → assemble standalone → deploy →
smoke-check `/api/health`. Manual trigger by default, matching
`deploy-mi-api.yml`.

---

## 8. Additions from the production-hardening pass

### 8.1 Pinned demo source (build-time trust boundary C, hardened)

`DEMO_SOURCE` in the generator names one dataset and asserts its identity —
client, portfolio, asset class, currency, reporting date, balance range,
minimum exposure count, and a SHA-256 of the canonical file — **before any
figure is computed**. The file fingerprint is checked before parsing, so a
substituted dataset fails diagnosably rather than as a downstream schema error.

There is no fallback: `mi_agent_api.data_source` is not imported, and a test
parses the module's AST to assert it never will be, and that the assertions are
unconditional (not behind a flag or a `try`).

`EXPECTED_DEMO_SOURCE` in `lib/config.ts` re-checks the same identity at
runtime against the pack's own `source` block, so a pack built from the wrong
portfolio cannot be served even if committed. Boundary C is now enforced from
both sides.

### 8.2 Startup configuration gate

`instrumentation.ts` runs once per server process, before any request. In
production it throws on any unsafe configuration; outside production it warns.
This converts a class of silent production defects — leads to a local file, a
forgeable session cookie, a localhost canonical URL, unlimited effective rate
limits — into a failed boot.

### 8.3 Rate-limit store

`RateLimitStore` has two adapters (in-memory, shared-over-HTTP) behind one
interface. The failure policy is asymmetric by design: **lead submission fails
closed**, demo queries **degrade to the stricter local limit**. The reasoning
is in `lib/rate-limit.ts` and the README; the short version is that refusing a
write is cheap and refusing to render the page is not.

### 8.4 Analytics collector

`POST /api/analytics` is an allow-list of ten event names with a fixed
property vocabulary, each value stripped to `[A-Za-z0-9_.-]` and capped. It
always returns 204 — a rejected event, a rate-limited caller and a malformed
body are indistinguishable, so the endpoint cannot be probed and analytics can
never shape the visitor's experience.

Campaign attribution is captured on arrival into `sessionStorage`, re-sanitised
server-side, attached to the lead notification and never echoed back.

### 8.5 Liveness versus readiness

`/api/health` answers "alive"; `/api/ready` answers "correctly configured", and
returns 503 with error codes when it is not. They are allowed to disagree, which
is the point — restarting a process fixes the first and never the second.

The App Service platform probe should point at `/api/health`: a probe on
`/api/ready` would turn a configuration error into a restart loop. `/api/ready`
belongs in deployment smoke tests, where a human can act on it.

### 8.6 Build-output scan

`scripts/scan-build.mjs` scans what actually ships. Two severities, because the
risks differ:

* **Secrets** — an error anywhere, server bundle included.
* **Internal Trakt references** (blob endpoints, the MI API host,
  `processed-v2/`, `MI_AGENT_*`, `TRAKT_COPILOT_*`) — an error anywhere, because
  their presence would mean this deployment reaches into infrastructure it is
  architected not to touch.
* **Build paths and server-only provenance** — an error only in client-visible
  artefacts. Next's standalone output records its own build directory and the
  pack's server-only provenance names its source dataset; neither is
  downloadable, and the client bundle is verified clean of both.
