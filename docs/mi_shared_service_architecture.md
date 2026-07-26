# The shared governed MI application service

React and Microsoft 365 Copilot are **presentation channels**. They share one
analytical implementation. This document is the architecture map, the service
contract, the root-cause analysis of what was diverging, and the deployment
steps.

---

## 1. Target architecture

```
                    ┌──────────────────────────────────────────────┐
  React MI Agent ──►│                                              │
   POST /mi/query   │   mi_agent_api.mi_service                    │
                    │   execute_governed_mi_query(MiQueryRequest)  │──► governed
  M365 Copilot   ──►│                                              │    envelope
POST /v1/copilot/   │   parsing · intent classification · dataset  │
      mi/query      │   resolution · metrics · dimensions ·        │
                    │   filters · deterministic calculation ·      │
                    │   top-N · point-in-time · temporal ·         │
                    │   cohort · forecast · risk · validation ·    │
                    │   reconciliation · provenance · artifacts    │
                    └──────────────────────────────────────────────┘
```

Both routes are thin adapters. Neither owns analytical behaviour, so an MI
capability added to the shared service is inherited by both channels at once.

### Before

```
React  ─► POST /mi/query ──┬─ chat_routing.try_route(frame_resolver=_resolve_run_dataframe)
                           ├─ _resolve_query_frame → run_mi_agent_query → adapt_workflow_result
                           └─ returns the React envelope
                                      ▲
Copilot ─► POST /v1/copilot/mi/query ─┘   calls app.query(QueryRequest(...))
                                          drops spec / validation / reconciliation /
                                          diagnostics / assumptions; no client context;
                                          no run/portfolio provenance; no text normalisation
```

`app.query` mixed HTTP concerns, analytical orchestration and React response
shaping in one FastAPI handler. Copilot reached into it as a function, which
made the *engine* shared but left the *contract* and the *inputs* divergent —
and left a run-id assumption that only the React UI happened to satisfy.

### After

```
React  ─► POST /mi/query ─────────────► mi_service.execute_governed_mi_query ─► envelope (verbatim)

Copilot ─► POST /v1/copilot/mi/query ─► mi_service.execute_governed_mi_query ─► envelope
                                        └─ Entra auth · client context · row cap · text normalisation
```

No HTTP loopback, no second parser, no second executor, no business logic in
`copilot_actions.py`.

---

## 2. Root-cause analysis

### 2.1 Copilot analytical divergence

`copilot_actions.ask_trakt_mi` called `app.query`, so the *engine* was already
shared — but three things diverged:

1. **Inputs.** Copilot passed only `question`, `portfolioId`, `asOfDate`. It
   never passed the deployment's client context, so client-scoped resolution
   (pipeline history, currency, portfolio provenance) defaulted to
   `client_001` regardless of the deployment.
2. **Outputs.** The Copilot schema carried `answer`, `interpreted`, warnings,
   source notes and a compact artifact extract. The parsed query spec,
   validation, reconciliation, diagnostics and assumptions — the fields that
   make an answer auditable — were dropped, so Copilot's answer could not be
   checked against React's.
3. **Boundary.** `app.query` was an HTTP handler, not a service. Any future
   change to its request/response shaping would silently change Copilot too, in
   ways nobody was testing.

**Fix.** `mi_agent_api/mi_service.py` — a channel-neutral
`execute_governed_mi_query(MiQueryRequest) -> dict`. Both routes call it. The
Copilot response now carries the full governed envelope.

### 2.2 Geography / LTV run-resolution failure

The failing string came from `chat_routing._route_geo`:

```python
if frame_resolver is None or not run_id:            # ← the bug
    return _envelope(..., answer="I can't resolve the funded book for a geographic view here.",
                     warnings=["insufficient-data: no funded frame for the run."])
```

`try_route` derives `(client_id, run_id)` from `portfolioId`. The React UI
always sends `"{client}/{run}"` from its portfolio selector, so `run_id` was
always present and the guard never fired. Copilot omits `portfolioId` for an
ordinary question, so `run_id` was `None` and **every** geographic question
failed — regardless of the data being present. `/health` reporting
`collateral_geography`, `current_loan_to_value`, `ltv_bucket` and 73 funded
loans was consistent with this: the data was fine, the route refused to look at
it.

A combined "regional concentration / LTV distribution" question routes to
`_route_geo` (it matches a geography term plus a concentration marker), which is
why the LTV half failed too; a standalone LTV question already reached the
point-in-time executor.

**Fix.** Geographic concentration is a point-in-time question:

* `_route_geo` no longer requires `run_id`;
* the frame resolver passed by the service is now
  `_resolve_query_frame("funded", pid)` — **exactly** the resolver the
  point-in-time executor uses — which returns the active platform canonical when
  no run is selected and the run's funded book when one is;
* `metadata.selectedRun` / `runRequired` report a run **only** where the
  analytical intent genuinely needed one (`_RUN_SCOPED_ROUTES`: temporal
  comparison, evolution, cohort progression, forecast, scenario, run-scoped risk
  and the funded bridge). `geo_exposure` is not in that set.

No Copilot-specific geography mapper and no Copilot-specific LTV bucketing were
introduced. Geography uses the governed field the semantic registry selects
(`collateral_geography`, ITL3-resolved via `geo.exposure_by_itl3`, postcode-derived
where the ITL3 field is absent) with its existing coverage/reconciliation
treatment. LTV uses the governed `ltv_bucket` dimension with its existing percent
scale, exclusion and null policies, ordering and labels.

### 2.3 Deck publication / discovery failure

`persist_investor_deck` existed and wrote `decks/{client}/latest|{period}/`, but:

* it was gated on `pptx_persist_enabled()`, which is **off** unless
  `AZURE_STORAGE_CONNECTION_STRING` / `TRAKT_BLOB_CONNECTION` is set or
  `TRAKT_INVESTOR_PPTX_PERSIST=true`. A run in that window generated the deck
  into its run directory and published nothing;
* there was **no mechanism to promote a deck after the fact**, so
  `orun_ere_20260703T224438/reports/investor_pack.pptx` stayed invisible to the
  deck resolver — `list_decks("ERE")` correctly reported none, and Copilot
  correctly said "No investor deck has been generated yet for ERE";
* the pointer carried only four fields (no checksum, size, run ids or generator
  version), the dated copy could be overwritten, and a replayed older run could
  regress `latest`.

**Fix.** A hardened publication contract plus an explicit backfill — see §5.

### 2.4 Rendering artefacts

The black squares are **Copilot's renderer**, not corrupted data. The MI Agent
emits characters the Copilot surface does not render: em/en dashes and the minus
sign in answers (`—`, `–`, `−`), the arrow in provenance and query-trace notes
(`→`, `⇒`), the Greek decorations on metric labels (`Δ`, `Σ`), relational
operators in warnings (`≥`, `≤`, `≈`), the middle dot separator (`·`), the
ellipsis (`…`), plus non-breaking spaces between a value and its unit,
zero-width joiners, private-use codepoints (e.g. `U+F0B7`, the Symbol-font
bullet) and variation selectors.

**Fix.** `mi_agent_api/copilot_text.py`, applied **in the Copilot adapter only**.
See §6.

---

## 3. Shared service contract

`mi_agent_api/mi_service.py`

### Input — `MiQueryRequest`

| Field | Meaning |
|---|---|
| `question` | The natural-language MI question (required). |
| `portfolio_id` | `"{client}/{run}"`, or a bare client id. Omit for the active dataset. |
| `as_of_date` | Optional as-of label. |
| `filters` | Caller-supplied filters (React drill-through). |
| `dataset_context` | `funded` \| `pipeline` \| `forecast`. Explicit wording in the question wins. |
| `context` | Free-form workspace context; `activeView` / `datasetContext` are read from it. |
| `source_portfolio_lens` | `total` \| `direct` \| `acquired` \| a cohort id. |
| `client_id` | Authenticated tenant/client context. Used only when no explicit portfolio was given — it never overrides one. |
| `options` | Channel-neutral execution options (reserved; no analytical effect). |

### Output — the governed envelope

| Key | Contents |
|---|---|
| `ok`, `error` | Governed success / controlled error state. |
| `question`, `interpreted` | Original and interpreted question. |
| `spec` | Parsed query specification: metric, aggregation, dimension(s), filters, top-N, ordering, chart type, temporal/forecast/risk mode. |
| `answer` | The engine's answer text. |
| `artifacts` | Analytical artifacts (kpi / table / chart / validation) — the supporting values. |
| `validation` | Deterministic validation result. |
| `reconciliation` | Coverage / reconciliation footer. |
| `sourceNotes` | Provenance. |
| `warnings`, `diagnostics`, `assumptions` | User-facing warnings, technical diagnostics, assumptions. |
| `queryTrace`, `dimensionInvariant`, `filterInvariant` | Parser → executor → renderer attribution and the fail-closed dimension/filter invariants. |
| `metadata.datasetContext` | Selected dataset (`funded` \| `pipeline` \| `forecast`). |
| `metadata.asOfDate` | Reporting date. |
| `metadata.selectedClient` / `selectedPortfolio` | Governed client and portfolio context. |
| `metadata.selectedRun` / `runRequired` | The run — **only** where the analytical intent genuinely required one; `null` / `false` otherwise. |
| `metadata.dataSourceKind` / `dataSourceLabel` | Governed data source (kind and label; never a path). |
| `metadata.route` | The analytical route taken, or absent for the point-in-time executor. |

### Ownership boundaries

| Owned by the shared service | Owned by the React adapter | Owned by the Copilot adapter |
|---|---|---|
| Parsing, intent classification, follow-up normalisation | Chart rendering | Entra bearer validation |
| Portfolio / dataset / active-dataframe resolution | Drill-through tables | Deployment client context |
| Metric, dimension, filter resolution | Workspace state | Reshaping into the Copilot OpenAPI schema |
| Deterministic calculation, top-N, point-in-time | Interactive components | Deterministic row truncation (flagged) |
| Temporal comparison, evolution, cohort, forecast, risk | | Unicode/Markdown normalisation |
| Validation, reconciliation, provenance, warnings | | Signed download URLs |
| Governed error handling, artifact creation | | Channel response-size limits |

The Copilot adapter must never reimplement routing, choose a different dataset,
define a metric, require a run for a point-in-time question, downgrade to a
simpler path, or replace an analytical failure with narrative.

### Data-resolution note

The concrete resolvers (`_resolve_query_frame`, `_resolve_run_dataframe`,
`_pipeline_history`, `_onboarding_output_root`, …) still live in
`mi_agent_api/app.py` next to the dashboard endpoints that share them.
`mi_service` imports them lazily so the only import direction is
`app` (HTTP) → `mi_service` (application) → `app` (data resolution, lazily).
Moving them is a separate, mechanical change; nothing about the service boundary
depends on where they live.

---

## 4. Files changed

| File | Change | Why |
|---|---|---|
| `mi_agent_api/mi_service.py` | **New.** `MiQueryRequest` + `execute_governed_mi_query` + `split_portfolio` + `_RUN_SCOPED_ROUTES`. | The single governed analytical entrypoint both channels call. |
| `mi_agent_api/app.py` | `POST /mi/query` reduced to a thin adapter; the orchestration body moved to `mi_service`; unused `_query_dataset_context` removed; `mi_service` imported. | React becomes a presentation channel. Envelope returned verbatim — contract unchanged. |
| `mi_agent_api/chat_routing.py` | `_route_geo` no longer requires `run_id`; degrades with a scope-accurate message when the frame is genuinely unavailable. | Geographic concentration is point-in-time and must answer from the active governed dataset. |
| `mi_agent_api/copilot_actions.py` | Calls `mi_service` directly; response widened to the full governed envelope; deterministic flagged truncation; text normalisation; tape client-mismatch guard. | Thin adapter with analytical parity and no credential/tenant leakage. |
| `mi_agent_api/copilot_text.py` | **New.** Channel-safe Unicode/Markdown normalisation. | Removes the black-square artefacts, in the Copilot channel only. |
| `mi_agent_api/decks.py` | `_latest_from_pointer` whitelists the richer pointer fields for both blob and local modes. | Surfaces period, as-of date, run ids, checksum, size and generator version without exposing storage paths. |
| `apps/blob_trigger_app/pptx_stage.py` | Write-once dated copy; no `latest` regression to an older period; pointer written last with checksum/size/content type/run ids/generator version; structural deck validation; `force` for the admin backfill; richer manifest artifact record. | A durable, idempotent, auditable publication contract. |
| `apps/blob_trigger_app/deck_backfill.py` | **New.** Discovery, governed selection, validation and promotion of un-published run decks, with dry-run and a CLI. | Makes already-generated decks (e.g. `orun_ere_20260703T224438`) discoverable. |
| `apps/blob_trigger_app/orchestrator_invoke.py` | Passes `source_run_id` into the PPTX stage. | Pointer provenance. |
| `deploy/copilot-agent/trakt-copilot-openapi.yaml` | New response fields; `askTraktMi` description tells Copilot to omit `portfolioId` for point-in-time questions. | Contract matches the widened response. |
| `deploy/copilot-agent/declarativeAgent.json` | Instructions updated: full capability list, omit `portfolioId`, report warnings and truncation. | Stops Copilot asking for a run id or extrapolating from truncated rows. |

Tests: `mi_agent_api/tests/test_mi_service.py`,
`mi_agent_api/tests/test_channel_parity.py`,
`mi_agent_api/tests/test_copilot_text.py`, `tests/test_deck_publication.py`,
`tests/test_deck_backfill.py`, plus additions to
`mi_agent_api/tests/test_copilot_actions.py` and an updated assertion in
`mi_agent_api/tests/test_decks.py`.

---

## 5. Deck publication and backfill

### Publication contract

Every successful orchestration run:

1. **retains** the immutable run artifact
   `out/_blob_trigger/{orchestration_run}/reports/investor_pack.pptx` — the
   publisher only ever reads it;
2. publishes the dated copy `decks/{client}/{YYYY-MM}/investor_pack.pptx`
   **write-once** — an existing dated publication is never overwritten by a
   different deck (the publication contract has no versioning);
3. replaces `decks/{client}/latest/investor_pack.pptx`, **unless** the current
   pointer names a newer reporting period — a replayed or late older run never
   regresses `latest`;
4. writes `decks/{client}/latest/latest_investor_pack.json` **last**, so a
   consumer never sees a pointer describing bytes that are not there yet.

The pointer carries: `client_id`, `reporting_period` (normalised `YYYY-MM`,
identical to the dated path), `as_of_date`, `generated_at`,
`orchestration_run_id`, `source_run_id`, `blob_name` / `period_blob_name`
(governed relative keys), `checksum` (`sha256:…`), `size_bytes`, `content_type`,
`generator`, `generator_version`. No credentials, account names or connection
strings.

The whole operation is idempotent and safe to retry.

### Backfill

```bash
# Plan only — changes nothing
python -m apps.blob_trigger_app.deck_backfill --dry-run

# Promote every un-published deck for one client
python -m apps.blob_trigger_app.deck_backfill --client ERE

# One specific run; supply the period when the manifest has none
python -m apps.blob_trigger_app.deck_backfill --run orun_ere_20260703T224438 --period 2026-07

# Machine-readable audit record
python -m apps.blob_trigger_app.deck_backfill --json
```

It scans **only** governed successful orchestration runs (`run_state.json`
status completed, no blockers, investor-pack artifact not `failed`), resolves
client and reporting period from the manifest, validates the file (non-empty
OOXML), computes the checksum, and publishes oldest-period-first so `latest`
ends on the newest valid period. Re-running is a no-op. Runtime Copilot
discovery never scans orchestration output — durable publication remains the
single source of truth.

Root scanned: `processed-v2/out/_blob_trigger` by default, overridable with
`--root` or `TRAKT_RUN_OUTPUT_ROOT`.

---

## 6. Unicode and Markdown normalisation

`mi_agent_api/copilot_text.py`, applied by the Copilot adapter to answers,
warnings, diagnostics, assumptions, source notes, artifact titles, KPI labels
and values, and string row cells. It is **not** applied to the React envelope,
the shared service, or any stored artifact.

Order of operations:

1. strip malformed citation remnants (`【…】`, `[[cite:…]]`, `[cite:…]`, `†source`);
2. apply an explicit symbol map — each replacement is an ASCII equivalent of the
   same meaning (`—`→`-`, `→`→`->`, `⇒`→`=>`, `≥`→`>=`, `≈`→`~`, `×`→`x`,
   `±`→`+/-`, `Δ`→`Delta`, `Σ`→`Sum`, `·`→`-`, `…`→`...`, smart quotes → ASCII
   quotes, `✓`→`[ok]`, `⚠`→`[!]`);
3. normalise with **NFKC** (the documented normalisation form; applied after the
   map so the characters that matter are handled deterministically and NFKC only
   folds the compatibility residue);
4. remove variation selectors, private-use codepoints, zero-width characters and
   C0/C1 controls (keeping `\t`, `\n`, `\r`);
5. fold non-breaking and other exotic spaces to an ordinary space;
6. tidy doubled whitespace per line, preserving Markdown structure.

Preserved by contract: `£` and other currency signs, `%`, decimal and thousands
punctuation, dates, ordinary bullets, useful Markdown — and **every numeric
value**. Numbers are never re-parsed or re-formatted; only strings are touched,
and dict keys are left alone.

---

## 7. Before / after

Client `ERE`, no `portfolioId`, no run id, via `POST /v1/copilot/mi/query`.

### Latest portfolio summary

| | Before | After |
|---|---|---|
| Result | 73 loans, £8.9m, source: platform canonical | unchanged — and now identical to React field-for-field |
| `selectedRun` | *(not reported)* | `null` (point-in-time) |
| `validation` / `querySpec` | *(absent)* | present |

```
ok: true | selectedClient: "ERE" | selectedRun: null
kpi: [{label: "Loan", value: "36"}, {label: "Current Outstanding Balance", value: "£5.4MM"}]
```

*(figures from the bundled demo tape used by the tests; the ERE deployment
returns 73 loans / £8.9m from its own governed canonical.)*

### Regional concentration

**Before**

```
answer:   "I can't resolve the funded book for a geographic view here."
warnings: ["insufficient data: no funded frame for the run."]
```

**After**

```
ok: true | selectedRun: null | route: geo_exposure
querySpec: {metric: current_outstanding_balance, dimension: collateral_geography,
            aggregation: sum}
answer: "Largest geographic concentration: Nottingham at £831k (15.4% of the book)
         across 10 ITL3 area(s). Basis: collateral; resolved coverage 100.0%."
rows:   [{area: Nottingham,         code: TLF14, balance: 830811.74, count: 5, share: "15.4%"},
         {area: Cambridgeshire CC,  code: TLH42, balance: 760586.78, count: 5, share: "14.1%"},
         {area: Leeds,              code: TLE42, balance: 751500.54, count: 5, share: "14.0%"}, …]
```

Identical rows and numbers in React.

### LTV distribution

**Before** — same failure as above when combined with a geography term.

**After**

```
ok: true | selectedRun: null
querySpec: {metric: current_outstanding_balance, dimension: ltv_bucket, aggregation: sum}
rows: [{ltv_bucket: "50-60%", current_outstanding_balance_sum: 1774558.49, loan_count: 10, concentration_pct: 32.97},
       {ltv_bucket: "40-50%", current_outstanding_balance_sum: 1711953.29, loan_count: 10, concentration_pct: 31.81},
       {ltv_bucket: "30-40%", current_outstanding_balance_sum: 1634455.43, loan_count: 13, concentration_pct: 30.37}, …]
```

The governed `ltv_bucket` dimension, not a channel-specific re-bucketing.

### Top 10 loans

```
ok: true | selectedRun: null | totalRows: 10 | truncated: false
rows: [{loan_identifier: "DEMO-0020", current_outstanding_balance: 273558.51,
        current_loan_to_value: 51.61, youngest_borrower_age: 82, …}, …]
```

Same ranking, values and filters as React.

### Latest investor deck

**Before**

```
404 — "No investor deck has been generated yet for ERE."
```

**After** (once the backfill has run)

```json
{ "ok": true, "artifactType": "investor_deck",
  "fileName": "ERE_investor_deck_latest.pptx",
  "contentType": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  "sizeBytes": 1843200, "clientId": "ERE", "reportingPeriod": "2026-07",
  "generatedAt": "2026-07-03T22:44:38+00:00",
  "downloadUrl": "https://…/v1/copilot/artifacts/download?token=…",
  "downloadExpiresAt": "2026-07-25T10:05:00+00:00" }
```

### Rendering

**Before** — `Balance is £8.9m ■ 73 loans`, `parser ■ executor ■ renderer`,
`■ balance −£1.2m`.
**After** — `Balance is £8.9m - 73 loans`, `parser -> executor -> renderer`,
`Delta balance -£1.2m`. No unrenderable codepoint survives a Copilot response
(asserted end-to-end).

---

## 8. Deployment

### Azure App Settings (`trakt-mi-api` App Service)

No **new required** settings. Existing settings still apply
(`TRAKT_COPILOT_ENTRA_TENANT_ID`, `TRAKT_COPILOT_ENTRA_AUDIENCE`,
`TRAKT_COPILOT_DOWNLOAD_SIGNING_KEY`, `TRAKT_COPILOT_DOWNLOAD_TTL_SECONDS`,
`TRAKT_COPILOT_PUBLIC_BASE_URL`, `MI_AGENT_CLIENT_ID`).

Recommended / optional:

| Setting | Where | Purpose |
|---|---|---|
| `MI_AGENT_CLIENT_ID` | MI API | Should be set to the deployment's client (e.g. `ERE`). Without it the client is inferred from `MI_AGENT_PLATFORM_URI`, then falls back to `client_001`. |
| `TRAKT_INVESTOR_PPTX_PERSIST` | orchestration | Set `true` to force durable deck publication. Auto-on when a blob connection is configured. **This being unset is why the existing ERE deck was never published.** |
| `TRAKT_RUN_OUTPUT_ROOT` | backfill CLI | Only if run output is not under `processed-v2/out/_blob_trigger`. |

### Migration steps

1. Deploy the MI API (code-only change; no schema or storage-layout migration).
2. Restart the App Service so the new router and service are loaded.
3. Set `MI_AGENT_CLIENT_ID` if it is not already set.
4. Confirm `TRAKT_INVESTOR_PPTX_PERSIST` / the blob connection so **future** runs
   publish their decks.
5. Run the backfill for the existing decks:
   ```bash
   python -m apps.blob_trigger_app.deck_backfill --dry-run --client ERE   # review
   python -m apps.blob_trigger_app.deck_backfill --client ERE             # promote
   ```
6. Verify: `GET /v1/copilot/artifacts/latest/investor-deck` returns metadata and
   a signed URL; the URL downloads the PPTX before expiry.

### Restart requirements

Yes — an App Service restart (or slot swap) is needed for the new modules.

### Copilot package ZIP

**Yes, regenerate and re-upload.** `trakt-copilot-openapi.yaml` (new response
fields, updated `askTraktMi` description) and `declarativeAgent.json`
(instructions) both changed:

```bash
python deploy/copilot-agent/package_agent.py   # → deploy/copilot-agent/dist/trakt-copilot-agent.zip
```

Then re-upload the app package in the Microsoft 365 admin centre / Teams
Developer Portal.

### OpenAPI schema

**Changed — additively.** New optional response properties on `CopilotMiAnswer`
(`querySpec`, `selectedClient`, `selectedPortfolio`, `selectedRun`, `validation`,
`reconciliation`, `diagnostics`, `assumptions`, `truncated`, `truncationNote`)
and `CopilotSupportingArtifact` (`totalRows`). No path, operation id, request
schema or required field changed, so an old package keeps working — it simply
ignores the new fields.

### Entra configuration

**Unchanged.** Same tenant, audience, scope, app registration and consent.

---

## 9. Parity statement

**Copilot now has analytical parity with the React MI Agent.** For the same
authenticated tenant, client, portfolio context, question and as-of date, both
channels invoke `execute_governed_mi_query` and return the same interpreted
intent, query specification, dataset, reporting date, filters, metric,
aggregation, dimensions, rows, numeric values, validation, reconciliation,
provenance, warnings and governed error state. This is asserted structurally
(not on prose) by `mi_agent_api/tests/test_channel_parity.py` over a
representative subset of the existing golden-question library.

Deliberate presentation-only differences:

* **Row cap.** Copilot caps supporting rows at 50; React is uncapped. The cap is
  deterministic (the first N rows in executor order — never re-ranked or
  re-aggregated) and explicitly reported via `truncated`, `truncationNote` and
  `totalRows`. Totals and percentages are always computed over all rows.
* **Charts.** React renders charts; Copilot receives the chart's underlying rows
  as supporting values.
* **Interactivity.** Drill-through, workspace state and follow-up UI are React
  only. Copilot rewrites follow-ups into standalone questions (both channels are
  stateless).
* **Text.** Copilot strings are Unicode-normalised for its renderer; the numbers
  are byte-identical.
* **Artifact links.** Copilot returns short-lived signed URLs; React uses its own
  authenticated download routes.

Remaining limitations:

* Analytical capability is bounded by the shared service. A question neither
  channel can answer is refused identically by both — Copilot does not
  compensate with narrative.
* Copilot has no `filters` or `sourcePortfolioLens` input, so React
  drill-through and lens selections have no Copilot equivalent. The shared
  service supports both; exposing them is a schema change, not an analytical one.
* A genuinely temporal question still needs its run/period context, supplied
  either in the question or via `portfolioId`.

---

## 10. Residual risks

| Risk | Assessment |
|---|---|
| **Backwards compatibility** | The React envelope is returned verbatim with additive `metadata` keys only. The Copilot schema is additive. An un-regenerated Copilot package keeps working. The one behaviour change is the deck pointer's `reporting_period`, now the normalised `YYYY-MM` key matching the dated path; the finer value moved to `as_of_date`. Old pointers written before this change still resolve — the fields are read defensively. |
| **Client isolation** | Copilot passes its deployment client into every query; the tape action fails closed on a platform-URI/client mismatch; download tokens carry the client inside the HMAC-signed payload, so it cannot be swapped. Deployment-per-client is unchanged and no client is hardcoded. |
| **Cache behaviour** | Unchanged. Both channels share the same in-process dataset, semantics and currency caches, so the second channel to ask a question pays no extra load. No new cache and no new invalidation path were introduced. |
| **Historical-run selection** | The rule is now explicit: a run is used when one is selected, or when the route is in `_RUN_SCOPED_ROUTES`. If a route is added that genuinely needs a run, it must be added to that set or it will silently answer point-in-time. The set is unit-tested. |
| **Latest-deck pointer integrity** | Ordering is by normalised period, so a run with an unparseable period cannot claim `latest` over a dated one. Concurrent publishers could still interleave the copy and the pointer write; the pointer-last ordering makes the worst case a pointer that briefly lags the bytes, never one that leads them. Blob-level conditional writes would close this fully. |
| **Response size** | The 50-row cap is a hard ceiling. A very wide table (many columns) can still be large; only row count is capped. Truncation is always flagged, and the declarative agent is instructed never to extrapolate a total from truncated rows. |
| **Backfill scope** | The backfill reads the run manifest, not the deck. A run whose manifest records the wrong client or period would publish under that wrong key. `--dry-run` exists precisely to review the resolved client/period before promoting. |
