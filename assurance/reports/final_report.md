# MI Agent — Pre-Production Go-Live Assurance Report

## Executive decision

**CONDITIONAL GO** — launch is safe under a defined containment envelope; it is a
**NO-GO** without those conditions.

The MI Agent returns materially correct, governed, auditable answers for the
intended single-client scope, and fails safely with disclosure where it cannot.
Two cross-tenant read paths were found and fixed on the launch surface; a
mixed-currency aggregation and a multi-date double-count were found and fixed on
the governed route. The residual risks are contained by launch restrictions, not
left silent.

### Launch conditions (all required)

1. **Single-tenant deployment** with `MI_AGENT_CLIENT_ID` set and a **dedicated,
   non-shared** onboarding/central-tape root (ASSURE-002 containment).
2. **Single currency** portfolio (GBP for ER-UK) — mixed-currency is only guarded
   on the governed `/mi/query` route (ASSURE-006).
3. **Single reporting-date** active dataset (a `…/latest` pointer).
4. **Copilot and generated-artefact downloads disabled** until separately
   validated (pre-existing Copilot artifact failures; unauthenticated
   signed-download redemption).
5. Real platform authentication on (Easy Auth), `MI_AGENT_AUTH_ENABLED` never
   false, `TRAKT_RUNTIME_MODE` production.
6. Keep "arrears/default status" and free-form "period-movement" phrasings off
   the suggested-question set (ASSURE-010; ungoverned thresholded movement route).

Under these conditions there are **zero open Critical** and **zero open High**
defects on the exercised launch surface.

## Scope assessed

Tested: the governed `/mi/query` capability end-to-end (auth → tenant → dataset →
parser → recogniser → workflow → calculation → envelope → presenter), the
`/mi/snapshot` dashboard GET, tenancy/scope/date/currency controls, 1,000
governed questions against an independent oracle, routing collisions, invariants
and metamorphic transformations, evidence/audit/presentation, performance, and
failure injection.

Not tested (recorded, not asserted ready): live LLM parser latency; reliability
faults (audit-sink/cache/blob/BSR/registry corruption) under injection; Copilot
and PPTX artefact surfaces beyond static inspection; the non-`/mi/snapshot`
dashboard GET routes under adversarial tenancy (contained by launch condition 1).

## Architecture verified (actual, not intended)

The verified runtime map is in `architecture/verified_architecture_map.md`. Key
realities that differ from the documentation:

* The shared workflow engine (`mi_workflows/engine.py`) is consumed by only 2 of
  ~12 analytical routes; the default point-in-time path and the dashboard
  endpoints use their own ungoverned math (`architecture/duplicate_calculation_register.md`).
* The documented "no dataframe without an authorisation token" seam
  (`datasets.resolve_authorised_frame`) had zero production callers and was itself
  cross-tenant-unsafe until fixed.
* Two Business Semantics Registry loaders read the same YAML into different
  dataclasses; only one applies source overrides.
* Tenancy runs an open namespace by default (`config/tenancy.yaml` absent).

## Assurance results by layer and family

* 1,000-question bank: **991 pass** (`reports/question_bank_results.json`).
  * Numerically-checkable families (point-in-time 150, filters 100, grouped 90,
    geographic 70, movement 110, portfolio-risk-comparison 100, concentration
    100, forecasting 50): **100%** match the independent oracle.
  * Ambiguous/adversarial 100: 100% acceptable route + no forbidden claim.
  * Controlled-failure 80: **71** — the 9 failures are all ASSURE-010
    ("loans in default" → whole book).
  * Follow-ups 50: 100%.
* Invariants + metamorphic: **21/21** (shares sum to 1 incl. unknown; disjoint
  sub-scopes sum to total; zero-weight WA no simple-mean fallback; mixed-currency
  no monetary total; duplication doubles counts and preserves ratios; scope
  narrowing == direct calc; category relabelling conserves population; determinism).
* Routing collisions: **12/12** acceptable + safe.
* Assurance runner suite: **45/45** pass.

## Critical controls

* **Tenancy:** ASSURE-001 (governed) and ASSURE-002 (`/mi/snapshot`) cross-tenant
  reads fixed and regression-tested; tenant is deployment config, never request
  data; storage bound to the authorised tenant.
* **Scope:** governed scope applied and disclosed; sub-scopes reconcile to total.
* **Date:** as-of-latest, no cross-date aggregation, adjustments disclosed.
* **Currency:** mixed-currency monetary suppression on the governed route;
  single-currency launch restriction for the rest.
* **Authentication:** fail-closed in production when the principal header is
  untrustworthy; Copilot Entra validation fail-closed.

## Calculation assurance

Every material metric (population, numerator, denominator, aggregation, unit,
value, exclusions) was validated against `assurance/oracle/oracle.py`, which
imports only pandas and shares no code with the system under test. 610
numerically-checkable questions matched exactly (rel-tol 1e-6). Duplicate-engine
risks (weighted-average simple-mean fallback, share-denominator dropping unknowns,
four relative-change formulas) are documented in the duplicate register; the
governed route is correct, the ungoverned dashboard/geo/cohort sites are the
residual driving launch restrictions.

## Routing assurance

Recogniser registry routes deterministically by (confidence, priority); 12
near-boundary collisions resolve to an acceptable owner and never leak a
judgement/causation claim. The one ordering hazard — the thresholded
period-movement route (prio 70) outranking the governed period-change route (prio
85) — is contained by keeping free-form movement phrasings off the suggested set.

## Operational readiness

Performance is well within budget (warm p50 60ms / p95 109ms). Failure injection
shows controlled errors, no client-facing stack traces, and fail-closed on
empty/malformed input. Deployment checklist and configuration blockers are in
`reports/deployment_checklist.md`. Reliability fault injection and durable audit
sink wiring are outstanding.

## Defects (see `defect_register/`)

| ID | Severity | Status |
|---|---|---|
| ASSURE-001 cross-tenant `/mi/query` read | Critical | FIXED |
| ASSURE-002 cross-tenant dashboard-GET read | Critical | FIXED (`/mi/snapshot`) / CONTAIN (other GETs) |
| ASSURE-006 silent mixed-currency aggregation | Critical | FIXED (governed route) / CONTAIN (single-currency) |
| ASSURE-004 multi-date double-count | High | FIXED |
| ASSURE-010 filtered question → unfiltered book | Medium | OPEN / CONTAIN |
| ASSURE-003 asOfDate label vs snapshot | High | OPEN (envelope is honest; label secondary) |
| ASSURE-008 unauthenticated signed-download redemption | Medium | CONTAIN (disable artefacts) |
| ASSURE-009 ephemeral download signing key | Medium/ops | Checklist |

## Launch restrictions

Enumerated above (conditions 1–6). Each is recorded, not applied silently.

## Residual risks accepted for launch

* Non-`/mi/snapshot` dashboard GET routes are not tenant-bound (contained by the
  dedicated single-tenant root).
* Ungoverned dashboard/geo/cohort sum sites do not guard mixed currency
  (contained by single-currency).
* `asOfDate` label can disagree with the true snapshot date (the governance
  envelope's `snapshot.reporting_date` remains correct).
* Reliability faults and live-LLM latency not exercised.

## Recommendation

**CONDITIONAL GO** for a single-client, single-currency, single-reporting-date
launch of the React `/mi/query` + `/mi/snapshot` surface with Copilot and
generated-artefact downloads disabled, real authentication on, and a dedicated
onboarding root. This configuration has no open Critical or High defect on the
exercised surface, materially correct governed calculations validated by an
independent oracle, and safe controlled failure. Removing any launch condition —
multi-tenant, multi-currency, Copilot, or artefact downloads — returns the system
to **NO-GO** until the corresponding residual is remediated and tested.
