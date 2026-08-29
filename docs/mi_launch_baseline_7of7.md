# MI Agent — frozen 7/7 launch baseline

**This is the marker every production change from here is measured against.**
Verified independently, by execution, before any go-live work began.

```
HEAD          879dd7765f631ee80d0960c12eefaa4995b97cdf
branch        claude/mi-query-agent-c7-2tlhr6
working tree  clean
parity        local == upstream == origin (879dd77)
```

## Frozen properties, each re-verified at this HEAD

| property | evidence | result |
|---|---|---|
| seven core routes semantically migrated | `analytical_meaning_census` | C1–C7 all 0 |
| post-claim raw-question semantic decision sites | census K1 | **0** |
| route-local semantic vocabularies | census K2 | **0** |
| no second owner across the governed concepts | census by concept + `semantic_owner_inventory` | none |
| fail-closed post-claim routing active | `route_substitution_detector` | **0 of 2** substituted |
| legitimate pre-claim fallback active | `test_failclosed_route_execution` F3/F3b | pass |
| ranked movement composes generically | `test_live_movement_receipt_evidence` | pass |
| ranked-movement live evidence path closed | same, incl. 6 mutation controls | pass |
| filtered ranked movement | `test_ranked_movement_filter_composition` | pass |
| no implicit measure / period defaults | `c7_independent_audit` check D | pass |
| D1 / D2 / D4 closed | frozen canary + audit | pass |
| MI-only assurance boundary established | `MI_REGRESSION_MANIFEST.txt` | 278 modules |
| unrelated workflows outside the MI denominator | boundary probes | OCC / onboarding / Annex 2 / mail **OUT** |

Guard suite at this HEAD: **61 passed** across the post-claim guard, the
fail-closed controls, the frozen canary bank, the live receipt evidence tests
and the ranked-movement filter composition tests.

Independent audit: **10 of 10**.

## Frozen canary baseline

```
INVARIANT BREACHES  0
UNEVIDENCED         21 declared elements across 9 cases (evidence gap, not breach)
UNEXERCISED         F3, F4, F5, F6, F7
```

**Canary observations are never rewritten.** A baseline that moves must carry a
ledger entry in `compound_canary_bank.yaml`; the bank may grow, it may not
shrink.

## MI regression denominator (authoritative)

```
manifest   migration_phase0/MI_REGRESSION_MANIFEST.txt   278 modules
baseline   5957 passed · 81 failed · 711 skipped · 15 xfailed · 4 errors
           1 timeout · 6768 executed · 85 failing/erroring names
```

OCC, onboarding, Annex 2, regulatory XML, mail and demo-platform suites are
**outside** this denominator, decided from the import graph rather than by
filename. A whole-repository run is a secondary integration check only and is
**NON-AUTHORITATIVE FOR MI COMMERCIAL GO-LIVE**.

## Residuals carried into launch (not defects)

1. Pre-claim recognition reads wording on **6 of 7** routes. Permitted; not
   redesigned here.
2. `concentration_analysis` and `portfolio_risk_comparison` still delegate the
   whole question to a workflow with its own recognition. Phase 3 addresses
   concentration; portfolio comparison is explicitly out of scope.
3. Evidence singularity is proven for ranked movement; other routes use the
   older receipt channel and were not re-measured.
