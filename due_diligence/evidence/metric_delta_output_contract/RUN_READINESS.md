# Output-contract canary — run readiness

Everything establishable without spending anything has been established, and it
is written down before the deploy rather than after, so nothing here is a reading
taken with knowledge of the outcome.

## Phase 1 — COMPLETE

The runner is built from the existing infrastructure and nothing else: the build
provenance reader, the `/mi/query` caller, the Kudu sink reader and poller, the
CandidateIntent / compiler / serving record readers and the corrected temporal
scorer are all reused, and the requested-metric judgement is imported from
`certification.py` rather than reimplemented. No new framework.

    RUNNER_SELF_TESTS = 7 / 7 PASS, in CI (run 34760496949, step 6)

| # | self-test | result |
| --- | --- | --- |
| 1 | the bank hashes to the pinned sha256 | PASS |
| 2 | the historical M03 shape is REJECTED | PASS |
| 3 | a communicated QUALIFIED answer is ACCEPTED | PASS |
| 4 | temporal presence: the receipt when present, else `time.stated` | PASS |
| 5 | a positive case served only by LEGACY_FALLBACK FAILS | PASS |
| 6 | the one-attempt guard names the file the workflow commits | PASS |
| 7 | the harness imported no product module | PASS |

Self-test 7 exists because of a defect of mine that the gate caught before any
spend. `certification._movement_tokens` imported the product's formatter to
render a movement; that pulls in the whole `mi_agent` package — pandas, yaml,
plotly — and this harness is stdlib-only on purpose. It passed on a workstation
that happened to have them installed for the regression runs and died on the
runner with `ModuleNotFoundError: yaml`. The rendered movement now comes from the
served envelope's own metric table, which the product built with the same
formatter in the same response, so there is no import and no second renderer.
Self-test 7 proves the property live — no `mi_agent*` in `sys.modules` after
everything else has run — rather than by a text scan a lazy import would satisfy.

## Measured free, 2026-09-13T13:40:37Z (run 34760496949)

| fact | value | how |
| --- | --- | --- |
| `DEPLOYED_SHA_NOW` | `5d84a6a31ccf0f255c6c9170de97b01532fa35da` | artefact stamp, `GET /` |
| `PRODUCT_SHA` to deploy | `5c436961ebe279bc0820ab006867b9f8a869bde2` | the certified SHA |
| product files changed since it | **none** | `git diff` outside `due_diligence/`, `.github/`, `tests/` |
| `BANK_SHA256` | `6465f4ed…` recomputed = pinned | sha256 |
| historical banks | v1–v4, the serving canary, the confirmation canary — all intact | sha256 vs each pin |
| `BANK_STATE` | **UNSPENT** | no committed result file |
| spend so far | model calls **0**, `/mi/query` calls **0** | the harness counts its own |

Provenance reads FAIL because `5c436961` is not deployed yet. That is the correct
answer before Phase 2, not a defect.

## Two blockers, both operator-held

### 1. `MI_AGENT_PLAN_SERVE` must be confirmed `off` before the deploy

    MI_AGENT_PLAN_SERVE_PRE_DEPLOY = UNREAD

It is an Azure App Service application setting and needs ARM access; this
repository holds `AZURE_MI_API_PUBLISH_PROFILE`, which grants Kudu and nothing
else. There is no free read of it from here.

**The honest expectation is that it is still `canary`.** It was set to `canary`
for the confirmation canary, and nothing since has returned it to `off` — the
post-run step that would have done so was reported at the end of that task and
never executed. That is precisely the state Phase 2 forbids at deploy time, so
this is a stop and not an assumption.

### 2. `MI_BEARER` has expired again

    TOKEN  expires 2026-09-13T13:27:34Z   expired = True   783s past

The third expiry in this programme, and the third time the free preflight caught
it before the bank. An Entra access token lives about an hour, so it must be
re-minted immediately before the run rather than in advance.

## The sequence from here

    1. operator   MI_AGENT_PLAN_SERVE = off               Portal
    2. ci         deploy 5c436961 from deploy/mi-api-5c436961
    3. ci         prove the running build IS 5c436961      GET /
    4. operator   MI_AGENT_PLAN_SERVE = canary  AND
                  re-mint MI_BEARER                        together, then tell me
    5. ci         the five, once, result committed by the job itself
    6. operator   MI_AGENT_PLAN_SERVE = off                Portal

Steps 2, 3 and 5 are `deploy-mi-api.yml` and `contract-canary.yml`. Steps 1, 4
and 6 are not reachable from CI and are not inferred.

`MI_AGENT_PLAN_SERVE = canary` is the literal word: `on`, `true` and `1` read as
OFF, and a service reading the flag as OFF answers every question by the legacy
path — which this bank scores as FAIL on every positive case, spending it for
nothing.

## On verifying step 6

The strongest verification available to CI of the flag being `off` would be a
`/mi/query` call, which costs an Opus interpretation this bank does not authorise
and which the instruction explicitly says not to spend on infrastructure. So the
post-run verification of `MI_AGENT_PLAN_SERVE = off` is the operator's Portal
reading, recorded as theirs. During the run itself the flag is verified
behaviourally and for free, from `serving.mode` and `principal_matched` on every
evidence record the run produces.

---

# Phase 2 — DEPLOYMENT CONFIRMED, 2026-09-13T14:00:16Z

Operator reading, Portal, before the deploy:

    MI_AGENT_PLAN_SERVE_PRE_DEPLOY = off

Deploy: run 34761151472, `deploy-mi-api.yml` dispatched on branch
`deploy/mi-api-5c436961`, whose head is the certified SHA and nothing else.
Succeeded 13:58 with no SCM-restart retry.

    DEPLOYMENT_REQUESTED_SHA = 5c436961ebe279bc0820ab006867b9f8a869bde2
    DEPLOYMENT_CONFIRMED_SHA = 5c436961ebe279bc0820ab006867b9f8a869bde2
    AUTH_STATUS              = PASS
    read                     = run 34761383364, step 8, FIRST attempt

HOW BOTH ARE PROVED WITHOUT QUOTING A LOG LINE. On a `workflow_dispatch` the
provenance step is fatal, and `provenance.py` returns 0 only when
`provenance == "PASS" and auth == "PASS"`; its provenance check is an equality
between the artefact stamp read from the running process and `--expect-commit`.
The step's exit code is therefore the comparison's own result, not a reading of
it. Step 8 succeeded, so the running service IS the certified SHA and the bearer
authenticated.

    SPEND SO FAR   model calls 0, /mi/query calls 0
    BANK_STATE     UNSPENT

## Still outstanding: Phase 3

    MI_AGENT_PLAN_SERVE = off  ->  canary

The bearer authenticated at 14:00:16Z and an Entra access token lives about an
hour, so the flag flip and the run should follow promptly. If more than about
forty-five minutes pass, `MI_BEARER` should be re-minted at the same time —
three of this programme's runs have been stopped by a token that aged out
between preparation and execution.
