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

* **ASSURE-002** (candidate High) — Dashboard GET routes (`/mi/snapshot`,
  `/mi/evolution/*`, `/mi/cohorts`, `/mi/geo/exposure`, `/mi/risk-limits`,
  `/mi/pipeline/*`, `/mi/forecast/*`) take `portfolioId`/`client_id`/`run_id`
  as query params and resolve frames with no tenancy authorisation, no
  source-approval check, and no audit. Same storage-path mechanism as
  ASSURE-001. Needs an adversarial test through the HTTP layer.
* **ASSURE-003** (candidate High) — `asOfDate` is a presentational label only on
  the point-in-time path: a caller can request an arbitrary date and receive the
  active dataset's numbers labelled as of that date, with no warning.
* **ASSURE-004** (candidate High) — Default `/mi/query` path aggregates across
  every reporting date present in a combined tape (no single-date guard), so a
  loan present at two cuts is summed twice. Governed workflows refuse this; the
  default route does not.
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
