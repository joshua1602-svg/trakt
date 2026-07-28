# MI Agent assurance — defect register

Live register. Severity per the programme's classification (Critical / High /
Medium / Low). "Status" is one of OPEN, FIXED, DISPOSITIONED (accepted with a
recorded reason), WONTFIX.

Fixes are minimal, isolated and regression-tested. Pre-existing failures are
never reclassified as harmless without verification, and expectations are never
weakened to make a test pass.

---

## ASSURE-001 — Cross-tenant portfolio read via caller-supplied portfolio_id

* **severity:** Critical
* **component:** `mi_agent_api/mi_service.py`, `mi_agent_api/datasets.py`
* **fixture:** synthetic multibook tape + a two-client onboarding root
* **question_id:** adversarial (`assurance/runners/test_tenancy_isolation.py`)
* **observed_result:** A caller authenticated as tenant `client_001`, posting
  `portfolio_id="client_002/2026_06"`, received `client_002`'s central tape
  (balance £9,999,990 marker) with `status: success`. The governance envelope
  stamped `tenant_id=client_001, portfolio_id=client_002/2026_06`.
* **expected_result:** The caller can only ever read its own tenant's data; a
  foreign selector must not redirect storage resolution.
* **root_cause:** `mi_service._run_analysis` discarded the `AuthorisedPortfolio`
  and re-parsed the raw `requested_portfolio_id`, using its first segment as a
  `client_id` for storage path resolution
  (`datasets._resolve_query_frame` → `_resolve_run_dataframe` →
  `snapshots.resolve_tape_path`, building `root/{client_id}/{run_id}/…`). With
  no `config/tenancy.yaml`, `trakt_core.tenancy` runs an open namespace that
  authorises any well-formed selector, so the cross-tenant string passed the
  gate and was then used verbatim as a client directory. The documented
  "no dataframe without an authorisation token" seam
  (`datasets.resolve_authorised_frame`) had zero production callers and was
  itself broken (it passed `authorised.portfolio_id`, i.e. the selector, as the
  client segment). The currency path (`_apply_request_currency`) had the same
  flaw and additionally cached the foreign currency under the caller's key.
* **production_impact:** Cross-tenant data exposure on any deployment whose
  `MI_AGENT_ONBOARDING_OUTPUT_ROOT` (or blob platform root) is shared across
  clients. Automatic go-live blocker.
* **fix:** Storage resolution is bound to the authorised tenant. New optional
  `tenant_id` on `datasets._resolve_query_frame` / `_apply_request_currency`
  (and `mi_service._resolve_frame`) overrides the client directory with the
  tenant; `mi_service._run_analysis` threads `authorised.tenant_id` into every
  frame and currency resolution; `resolve_authorised_frame` now binds to
  `authorised.tenant_id`. The caller's `run_id` still selects the dated cut, and
  within-tenant book narrowing continues via the governed lens — only the client
  directory is forced to the tenant. Dashboard GET callers that pass no
  `tenant_id` are unchanged (they are separately tracked under ASSURE-002).
* **tests_added:** `assurance/runners/test_tenancy_isolation.py` (4 cases:
  foreign portfolio refused, legitimate own run resolves, default resolves
  active dataset, envelope never certifies a foreign balance). Test doubles in
  `mi_agent_api/tests/test_mi_service.py` updated to the new signature.
* **regression:** `tests/test_governance_context_and_tenancy.py`,
  `tests/test_governance_source_policy.py`, `mi_agent_api/tests/test_mi_service.py`,
  `test_currency.py`, `test_mi_query_lens_matrix.py`, `test_channel_parity.py`,
  `test_mi_query_route_contract.py` — 224 passed.
* **commit:** (this change)
* **status:** FIXED

---

## Candidate findings pending verification (from Phase 1 discovery)

Tracked here so nothing is lost; each is verified before classification.

## ASSURE-002 — Cross-tenant read on the dashboard GET path

* **severity:** Critical
* **component:** `mi_agent_api/app.py` dashboard GET routes
* **fixture:** two-client shared onboarding root (`assurance/runners/test_tenancy_isolation.py`)
* **observed_result:** `GET /mi/snapshot?client_id=client_secret&run_id=2026_06`
  on a deployment configured `MI_AGENT_CLIENT_ID=client_001` returned
  client_secret's tape (£8.9MM marker), HTTP 200. The dashboard GET routes take
  `client_id`/`run_id`/`portfolioId` as query params and call
  `_resolve_run_dataframe(client_id, run_id, root)` directly — no
  `authorise_portfolio_access`, no tenant binding, no source-approval, no audit.
* **expected_result:** A dashboard GET can only read the deployment tenant's data.
* **root_cause:** Same mechanism as ASSURE-001, on the GET path, which never
  routes through the governed capability.
* **production_impact:** Cross-tenant read on any deployment with a shared
  multi-client onboarding root. React calls `/mi/snapshot`, so it is on the
  launch surface.
* **fix (primary route):** `/mi/snapshot` now binds the storage `client_id` to
  `default_tenant_id()` (the deployment tenant); the caller's `run_id` still
  selects the cut. Verified: `?client_id=client_secret` no longer leaks; legit
  same-tenant request unchanged.
* **residual:** `/mi/evolution/*`, `/mi/cohorts`, `/mi/geo/exposure`,
  `/mi/risk-limits`, `/mi/pipeline/*`, `/mi/forecast/*` share the pattern and are
  NOT yet bound. A full sweep is a broader change than a minimal isolated fix.
* **containment:** Launch on a single-tenant deployment with a **dedicated**
  (non-shared) onboarding root, so no other tenant's tape exists under the root.
  This is a hard go-live condition; multi-tenant requires binding all GET routes.
* **tests_added:** `test_tenancy_isolation.py::test_snapshot_get_cannot_read_another_tenants_tape`.
* **status:** FIXED (`/mi/snapshot`) / CONTAIN (other GET routes; single-tenant + dedicated root)

## ASSURE-010 — Filtered question answered with the unfiltered population

* **severity:** Medium
* **component:** default point-in-time route (`mi_query_executor`)
* **fixture:** `three_portfolios` (has no default flag; `default_date` empty)
* **question_id:** `cfl_*` "how many loans are in default?" (9 variants)
* **observed_result:** "How many loans are in default?" returned 118 (the whole
  book), `status: success`, no disclosure — the "in default" qualifier was
  dropped (there is no governed default field, so the filter silently vanished).
* **expected_result:** Either a governed refusal ("default status not reported")
  or an explicitly zero/degraded answer — never the unfiltered book presented as
  the answer.
* **root_cause:** An unrecognised qualifier that maps to no governed field is not
  applied and not disclosed; the executor answers the base metric on the full
  population. Contrast "arrears" which maps to a present `arrears_balance` column
  and correctly returns £0.
* **production_impact:** A reader could infer 118 loans are in default. Narrow
  (only bites when a filter term has no backing field) but genuinely misleading.
* **fix:** Not fixed (would touch the frozen parser/executor filter-recognition
  path; larger than a minimal isolated change). Recorded for remediation.
* **status:** OPEN (recommend containment: keep default/arrears status off the
  suggested-question set until filter-recognition degrades closed with disclosure)

### Remaining candidates pending verification
* **ASSURE-003** (candidate High) — `asOfDate` is a presentational label only on
  the point-in-time path: a caller can request an arbitrary date and receive the
  active dataset's numbers labelled as of that date, with no warning.
## ASSURE-004 — Default point-in-time route aggregates across reporting dates

* **severity:** High
* **component:** `mi_agent_api/mi_service.py`
* **fixture:** combined two-date multibook tape (2026-05-31 + 2026-06-30)
* **question_id:** `assurance/runners/test_reporting_date_safety.py`
* **observed_result:** "What is the total outstanding balance?" against a
  combined tape returned £73,985,637 over 234 records — the sum of *both*
  cut-off dates — labelled `reporting_date=2026-05-31` (first row), with no
  warning. Every loan present at both cuts was counted twice.
* **expected_result:** A point-in-time KPI resolves as of the latest reporting
  date (£37,270,061, 118 records) and never silently aggregates incompatible
  dates.
* **root_cause:** The default point-in-time path (`run_mi_agent_query` →
  `mi_query_executor`) has no reporting-date axis; `_resolve_query_frame`
  returns the whole active frame when no run is selected. The governed
  comparison/concentration workflows refuse a multi-date scope; the default
  route did not.
* **production_impact:** Materially wrong KPI whenever the active dataset is a
  combined multi-date tape. Standard `…/latest` blob pointers resolve to a
  single cut, so the common wiring is unaffected, but combined tapes are a
  documented shape (`datasets._platform_runs`).
* **fix:** `_narrow_to_latest_reporting_date` narrows the point-in-time frame to
  its latest cut when multiple dates are present and discloses the narrowing as
  a warning. No-op for single-date frames and for the reporting-date-less
  in-memory fixtures, so existing behaviour is byte-identical in the common case.
* **tests_added:** `assurance/runners/test_reporting_date_safety.py` (3 cases:
  as-of-latest not aggregated, narrowing disclosed, single-date unchanged/unwarned).
* **status:** FIXED

## ASSURE-006 — Silent mixed-currency monetary aggregation

* **severity:** Critical
* **component:** `mi_agent_api/mi_service.py` (default point-in-time route); ungoverned sum sites across `mi_agent/mi_query_executor.py`, `mi_agent_api/{snapshots,evolution,cohorts,geo,movement_summary}.py`
* **fixture:** `assurance/fixtures/generated/mixed_currency.csv` (GBP + EUR)
* **question_id:** `assurance/runners/test_currency_safety.py`
* **observed_result:** On a GBP+EUR book, "total outstanding balance" returned
  £37.3MM — the arithmetic sum across both currencies — formatted with the modal
  (£) symbol, `status: success`, no warning. The independent oracle suppresses
  the monetary total for the same population.
* **expected_result:** Monetary totals over a mixed-currency population are
  suppressed (not a single figure) with a disclosed limitation; count-based
  measures continue.
* **root_cause:** Only `mi_workflows/engine.py` and
  `mi_agent/period_change/calculations.py` implement a mixed-currency guard;
  9 of 11 sum implementations — including the default `/mi/query` point-in-time
  path and every dashboard KPI/evolution/geo/cohort endpoint — add monetary
  values with no guard. `currency.resolve_currency_code` selects the modal
  currency and formats the cross-currency sum with that symbol.
* **production_impact:** Materially wrong, misleadingly-labelled monetary totals
  on any multi-currency book. Silent mixed-currency aggregation is a Critical /
  automatic no-go for a multi-currency launch.
* **fix (partial, primary route):** `_currency_limitation` detects a
  mixed-currency point-in-time population and `_suppress_monetary_values` blanks
  monetary KPI + reconciliation figures with a disclosed limitation and a
  `currencyLimitation` metadata flag. No-op for single-currency frames (the base
  fixture and standard single-currency deployments are byte-identical).
* **residual:** The guard covers the governed `/mi/query` point-in-time route.
  The dashboard GET routes and geo/cohort surfaces share the ungoverned sum sites
  and are NOT covered. Full remediation is a shared-calculation consolidation
  beyond a minimal isolated fix.
* **tests_added:** `assurance/runners/test_currency_safety.py` (suppression,
  disclosure, count-measure still answered, single-currency unchanged).
* **status:** FIXED (primary route) / CONTAIN (launch recommended single-currency
  until dashboard + geo/cohort sum sites are remediated)

### Remaining candidates pending verification
* **ASSURE-005** (candidate Critical) — Weighted averages silently fall back to
  a simple mean on zero total weight in `analytics_lib/stratify.py` and
  `mi_agent_api/snapshots.py`, publishing an unweighted number under a
  "weighted average" label with no flag.
* **ASSURE-006** (candidate Critical) — Monetary sums add across currencies with
  no guard on the default `/mi/query` path and the dashboard endpoints; only the
  two governed workflows guard mixture. `currency.resolve_currency_code` labels a
  mixed-currency total with the modal currency symbol.
* **ASSURE-007** (candidate High) — Share denominators drop unknown/unresolved
  rows in `geo.py` (share of resolved balance, not of the book), and via
  `missing_policy="exclude"` in the executor, overstating concentration.
* **ASSURE-008** (candidate Medium) — Copilot signed-download redemption
  (`/v1/copilot/artifacts/download`) has no auth dependency; tenant is read from
  the token payload. Control is HMAC integrity + 300s TTL only.
* **ASSURE-009** (candidate Medium/ops) — `TRAKT_COPILOT_DOWNLOAD_SIGNING_KEY`
  unset ⇒ ephemeral per-process key (warning only); breaks multi-worker token
  redemption. Deployment-checklist item.
