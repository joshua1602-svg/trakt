# Assurance metrics (§33)

No single aggregate percentage is presented as proof of readiness — the
containment envelope, not the headline, is the safety statement.

## Question bank execution

| Metric | Value |
|---|---|
| Total questions executed | 1,000 / 1,000 |
| Passed | 991 |
| Failed | 9 (all ASSURE-010 "loans in default" → unfiltered book) |
| Routing accuracy (acceptable owner) | 1,000 / 1,000 |
| Structured-value accuracy (numerically-checkable, vs independent oracle) | 610 / 610 (100%) |
| Scope accuracy (point-in-time + filters + grouped, oracle) | 340 / 340 |
| Reporting-date accuracy | as-of-latest enforced; multi-date disclosed (ASSURE-004 fixed) |
| Controlled-failure accuracy | 71 / 80 |
| Presentation accuracy (no forbidden claim on governed routes) | 1,000 / 1,000 |
| Deterministic-repeatability | identical structured output on repeat (invariant test) |

## Supporting suites

| Suite | Result |
|---|---|
| Invariant tests (§22) | 13 / 13 |
| Metamorphic tests (§23) | 8 / 8 |
| Routing collision matrix (§21) | 12 / 12 |
| Tenancy isolation (governed + snapshot GET) | 5 / 5 |
| Reporting-date safety | 3 / 3 |
| Currency safety | 4 / 4 |
| **Assurance runner total** | **45 / 45** |

## Latency

| Path | p50 | p95 | max |
|---|---|---|---|
| Cold | 150 ms | 212 ms | 546 ms |
| Warm | 60 ms | 109 ms | 111 ms |

## Defects by severity

| Severity | Fixed | Open/Contained |
|---|---|---|
| Critical | 3 (ASSURE-001, 002, 006) | 0 open (002/006 residuals contained) |
| High | 1 (ASSURE-004) | 1 (ASSURE-003, envelope honest) |
| Medium | 0 | 3 (ASSURE-008, 009, 010; contained) |

## Tests not executed and why

* Live LLM parser latency — no `ANTHROPIC_API_KEY` in the assurance environment.
* Reliability fault injection (audit-sink / cache / blob / BSR / registry
  corruption) — deferred to a pre-multi-tenant reliability sweep.
* Non-`/mi/snapshot` dashboard GET adversarial tenancy — contained by the
  single-tenant dedicated-root launch condition rather than fixed.
* Copilot / PPTX artefact surfaces — recommended disabled at launch; 9
  pre-existing Copilot artifact tests fail on the branch base.

## Pre-existing (non-assurance) failures

13 MI-suite failures pre-exist on the branch base `accb9b9` (verified on a clean
worktree); documented in `reports/baseline_test_run.md`, not fixed (no unrelated
repairs). None is a cross-tenant, currency, or materially-wrong-number defect.
