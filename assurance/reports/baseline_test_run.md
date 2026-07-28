# Baseline test run

Established before remediation, then re-confirmed after each fix. The full
repository suite (4,736 collected) includes the securitisation engine,
onboarding and landing page, which are out of MI-assurance scope; the
MI-relevant baseline below is the suite that exercises the MI Agent.

## MI-relevant suite

Command:
```
python -m pytest mi_agent/tests mi_agent_api/tests tests/mi_agent_pptx \
    mi_workflows assurance/runners -q
```

Result (post-fix, with the assurance runners included):
**1845 passed · 13 failed · 1 skipped · 20 xfailed** (≈171s).

The 7 new assurance tests all pass. The 2 updated `test_mi_service` doubles
pass. The 13 failures are **pre-existing on the branch base `accb9b9`** —
verified by re-running each on a clean worktree at that commit, where all 13
reproduce identically. My changes introduce zero new failures.

## Pre-existing failures — verified, not fixed (out of scope)

The programme forbids repairing unrelated repository issues to improve headline
counts, so these are documented rather than fixed. Each was root-caused to
confirm it is not masking an MI-safety defect.

| Test | Root cause | MI-safety relevance |
|---|---|---|
| `test_copilot_actions.py` — 9 tests (deck/tape/registry resolve + signed download scope) | Copilot artifact-type registry drift: a `canonical_tape` request resolves to the investor-deck resolver and returns the deck's "no investor deck available" message; the download-scope assertions depend on that path | Copilot artefact surface only (Priority 3). Recommends launch containment: disable Copilot + generated-artefact download until separately validated |
| `test_funded_enrichment.py::test_all_core_dimensions_available` | `borrower_type` dimension not derived because its derivation source column is absent from the enrichment fixture | Data-coverage gap on one dimension in one fixture; the agent reports the dimension as unavailable rather than fabricating — a controlled degrade, not a wrong answer |
| `test_portfolio_risk_comparison_route.py::test_existing_route_order_is_preserved` | Stale expectation: the asserted route list predates `period_change_analysis` registration (index-8 drift) and the `evolution` route | Test drift; the live registry is internally consistent and covered by other tests |
| `test_recogniser_registry.py::test_lens_aware_routes_are_declared_on_the_recogniser` | Same registry drift after new routes were added | Test drift |
| `test_funded_central_tape.py::test_health_reports_funded_source` | `/health` response shape changed (`dataSourceInfo` key relocated) | Health-endpoint contract drift; not an answer-path defect |

## Disposition

None of the 13 pre-existing failures is a cross-tenant, currency, or
materially-wrong-number defect. The Copilot artefact failures are the most
material and directly support the launch recommendation to contain Copilot and
generated-artefact downloads until they are separately validated. The route-order
and health-shape failures are stale tests that should be refreshed by the owning
team (outside this assurance scope, which must not weaken or rewrite unrelated
expectations).
