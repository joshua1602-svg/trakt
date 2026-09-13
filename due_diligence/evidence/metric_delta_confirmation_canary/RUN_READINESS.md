# Confirmation canary — run readiness

Everything that could be established without spending anything has been, and it
is recorded here before the deploy rather than after, so nothing in it is a
reading taken with knowledge of the outcome.

## Measured, 2026-09-13T11:50:33Z (run 34755479012, registration push)

| Fact | Value | How it was established |
| --- | --- | --- |
| `DEPLOYED_SHA_NOW` | `88cf5fde759c29f1cff29b9adbf003a2d6ebd530` | `GET /` → `build_info.json` stamp, read out of the running process |
| `PRODUCT_HEAD_TO_DEPLOY` | `5d84a6a31ccf0f255c6c9170de97b01532fa35da` | `git rev-parse` |
| `ADDITIONAL_PRODUCT_COMMIT_RIDING` | none | `git diff 88cf5fde..5d84a6a3` outside `due_diligence/`, `.github/`, `tests/` is exactly `plan_material_summary.py`, `plan_metric_delta.py`, `plan_serving_canary.py` |
| `BANK_SHA256` | `9705705c7c66a605ccd3f16a08275b33f1b05b201c5f09f99bcd24386eb7d065` | recomputed and matched against the committed pin |
| `HISTORICAL_BANKS` | v1, v2, v3, v4 and the spent serving canary — all intact | sha256 against each committed pin |
| `SCORER_SELF_TEST` | PASSED 4/4 | both directions of the S07/S08 temporal fault |
| `MI_BEARER` | valid now | `GET /mi/catalogue` → 200 |
| `SPEND_SO_FAR` | model calls 0, `/mi/query` calls 0 | the harness counts its own calls |
| `BANK_STATE` | UNSPENT | no result file is committed |

The provenance line reads FAIL, and that is the correct answer: the service is
still running the previous build because the deploy has not happened. The step is
non-fatal on the registration push for exactly that reason, and fatal on the
dispatch that spends the bank.

## Not measured, and not guessable

    MI_AGENT_PLAN_SERVE_PRE_DEPLOY = UNREAD

`MI_AGENT_PLAN_SERVE` and `MI_AGENT_PLAN_SERVE_PRINCIPALS` are Azure App Service
APPLICATION SETTINGS. Reading them needs ARM access to `trakt-mi-api`; this
repository's mi-api automation holds `AZURE_MI_API_PUBLISH_PROFILE`, which grants
Kudu and nothing else. There is no free read of that value from here.

It was set to `canary` for the previous live run and nothing in this repository
has changed it since, so the honest prior is that it is still `canary` — which is
precisely the state the operator's own sequencing says must not be true when the
new build goes out. That is why this is a stop and not an assumption.

## The two operator steps, and where each sits in the order

    1.  operator   MI_AGENT_PLAN_SERVE = off          Portal, before the deploy
    2.  ci         deploy 5d84a6a3                    deploy-mi-api.yml on
                                                      deploy/mi-api-5d84a6a3
    3.  ci         prove the running build IS 5d84a6a3
    4.  operator   MI_AGENT_PLAN_SERVE = canary       Portal, after step 3
                   MI_AGENT_PLAN_SERVE_PRINCIPALS unchanged — the exact oid,
                   no wildcard
    5.  ci         the five, once, and the result committed by the job itself

Steps 2, 3 and 5 are `metric-delta-canary.yml` and `deploy-mi-api.yml`. Steps 1
and 4 are not reachable from CI and are not inferred.

`MI_AGENT_PLAN_SERVE = canary` is the literal word. `on`, `true` and `1` are read
as OFF by `plan_serving_canary.handles()`, and a service that reads the flag as
OFF answers every question by the legacy path — which this bank scores as FAIL on
all five, spending it for nothing.
