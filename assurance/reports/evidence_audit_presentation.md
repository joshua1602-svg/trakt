# Evidence, audit and presentation review (Phase 7)

Verified against the live governed `/mi/query` capability on the base fixture.

## Evidence completeness (§24)

Every successful material answer carries a `reconciliation` evidence block and a
governance envelope sufficient to reproduce the result:

* dataset identity + `snapshot_id` (content-hashed, e.g. `three_portfolios@cba41e085db8`)
* tenant (`ExecutionContext.tenant_id`, from trusted context)
* portfolio scope (`ScopeRef`: portfolios in scope / used / excluded, field
  coverage per portfolio, consolidation disclosure)
* reporting date (`SnapshotRef.reporting_date`, 2026-06-30)
* population (`total_records`), filters applied, `records_after_filters`,
  `records_included`, `records_excluded_missing`
* numerator/denominator basis (`balance_included`, `balance_field`,
  `coverage_by_balance_pct`), missing-dimension and missing-measure policy
* currency treatment (suppression flag + limitation when mixed)

Evidence does not expose another tenant's data (confirmed by the tenancy fixes;
scope disclosure lists only the authenticated tenant's portfolios).

**Verdict: PASS** for the governed `/mi/query` path.

## Audit completeness (§25)

`trakt.audit` events captured directly for three cases:

| Case | outcome | tenant | error_code | capability | snapshot |
|---|---|---|---|---|---|
| Success | success | client_001 | – | mi.question.answer | three_portfolios@… |
| Unsupported ("NNEG") | error | client_001 | UNSUPPORTED_QUESTION | mi.question.answer | three_portfolios@… |
| Foreign client_id | blocked | client_001 | TENANT_MISMATCH | mi.question.answer | – |

Each event carries the effective tenant, capability, outcome, error code and
(where applicable) snapshot id; the forbidden-key scrub is applied; auditing
never raises.

**Verdict: PASS** for `/mi/query`. **Residual:** dashboard GET routes and
auth-layer 401/403 emit no audit events (tracked with ASSURE-002; contained by
the single-tenant launch condition).

## Presentation fidelity (§26)

* Rendered answers disclose scope, reporting date, units, aggregation, share
  basis, unknown population and comparability limitations (reconciliation +
  scope disclosure + warnings).
* Mixed-currency limitation is surfaced as a warning and the monetary KPI is
  suppressed (ASSURE-006 fix).
* Reporting-date narrowing is disclosed (ASSURE-004 fix).
* Banned judgement/causation wording — *safer, within appetite, compliant,
  excessive, primarily driven by, breaches appetite* — scanned on a portfolio
  comparison answer: **none present**. The governed comparison and concentration
  workflows are observational only ("not an overall assessment", "never a
  judgement").

**Verdict: PASS** for the governed workflows. **Residual (ASSURE — presentation
layer):** the ungoverned `movement_summary`/`chat_routing` period-movement route
carries hardcoded materiality thresholds and can assert causation ("primarily
driven by completions"). It is a lower-priority route (prio 70) that fires before
the governed period-change route (prio 85); recommend containment — keep
period-movement off the suggested-question set until it routes through the
governed workflow. The 1,000-question movement family (110) passed the
forbidden-claim check because those questions resolved to the governed route,
but a differently-phrased movement question can still reach the thresholded
route.
