# Go-live scorecard

Assessment: **PASS** / **PASS WITH CONDITIONS** / **FAIL** / **NOT TESTED**.
Any Critical failure or any untested onboarding-critical path ⇒ NO-GO.

| # | Gate | Assessment | Basis |
|---|------|-----------|-------|
| 1 | Tenant isolation | **PASS WITH CONDITIONS** | ASSURE-001 (governed) fixed + tested; ASSURE-002 fixed on `/mi/snapshot`, other GET routes contained by single-tenant + dedicated onboarding root |
| 2 | Dataset correctness | **PASS WITH CONDITIONS** | Env-driven resolution correct; must not use the synthetic glob fallback; single-date cut required (checklist) |
| 3 | Scope correctness | **PASS** | 350/350 point-in-time + filter + grouped questions match the oracle under governed scope; disjoint sub-scopes sum to total (invariant) |
| 4 | Reporting-date correctness | **PASS** | ASSURE-004 fixed: as-of-latest, no cross-date aggregation, disclosed |
| 5 | Currency safety | **PASS WITH CONDITIONS** | ASSURE-006 fixed on `/mi/query`; dashboard/geo/cohort sum sites unguarded ⇒ single-currency launch restriction |
| 6 | Calculation accuracy | **PASS** | 610/610 numerically-checkable questions match the independent oracle exactly; 21 invariant + metamorphic tests pass |
| 7 | Routing accuracy | **PASS** | 12 collision cases route to an acceptable owner with no forbidden-claim leak; 460 routed-family questions pass |
| 8 | Controlled failure | **PASS WITH CONDITIONS** | 71/80 refuse/degrade correctly; ASSURE-010 (a filtered question answered with the unfiltered book for a fieldless qualifier) open — keep such questions off the suggested set |
| 9 | Evidence completeness | **PASS** | Reconciliation + scope + snapshot evidence reproduces every material answer; no cross-tenant exposure |
| 10 | Audit completeness | **PASS WITH CONDITIONS** | `/mi/query` audits success/controlled-failure/blocked with effective tenant; dashboard GETs unaudited (contained) |
| 11 | Presentation fidelity | **PASS WITH CONDITIONS** | Governed workflows observational, no banned wording, limitations disclosed; ungoverned period-movement route can assert causation ⇒ contain |
| 12 | API reliability | **PASS** | Controlled errors, no client stack traces, fail-closed on empty/malformed |
| 13 | React compatibility | **PASS** | Single scoped query builder, tenant never client-sent, governed envelope preserved |
| 14 | Copilot compatibility | **FAIL → DEFER** | 9 pre-existing Copilot artifact tests fail; signed-download redemption has no auth dependency. Disable at launch |
| 15 | Artefact security | **PASS WITH CONDITIONS** | Governed capability tenant-binds deck access; signed-download signing key + redemption auth need work ⇒ defer generated artefacts |
| 16 | Deployment configuration | **PASS WITH CONDITIONS** | Checklist produced; example app settings incomplete; fail-closed posture correct |
| 17 | Operational monitoring | **PASS WITH CONDITIONS** | Audit logger present; durable sink + GET-route audit outstanding |
| 18 | Performance | **PASS** | Cold p50 150ms/p95 212ms, warm p50 60ms/p95 109ms; no timeout risk |

## Untested paths required for onboarding

* Live LLM parser latency (no API key) — deterministic path covers the common case.
* Reliability faults not injected: audit-sink / cache / blob / BSR / registry
  corruption.
* Copilot and generated-artefact surfaces — recommended disabled at launch.

## Decision drivers

* No **open Critical** on the recommended launch configuration (single tenant,
  dedicated onboarding root, single currency, Copilot/artefacts disabled).
* Both cross-tenant reads (ASSURE-001, ASSURE-002) fixed on the launch surface
  and regression-tested; the residual GET routes are contained by the launch
  conditions.
* All numerically-checkable answers match an independent oracle.
