# Deployment, provenance, and the preflight that stopped the canary

```
OPERATOR_STATE_AT_START   MI_AGENT_PLAN_SERVE = off   (checked manually in the
                          Azure Portal on trakt-mi-api; treated as authoritative)

DEPLOYMENT_SHA            88cf5fde759c29f1cff29b9adbf003a2d6ebd530
CURRENT_DEPLOYED_SHA      88cf5fde759c29f1cff29b9adbf003a2d6ebd530   CONFIRMED
MI_AGENT_PLAN_SERVE       off — UNCHANGED, across the deployment and after it
AUTH_PATH_READY           NO  — the stored bearer has expired
MI_BEARER_AVAILABLE       YES — present, 1826 characters

LIVE_MODEL_CALLS          0
LIVE_MI_QUERY_CALLS       0
CANARY_QUESTIONS_ASKED    0   (of 10 authorised; the bank is unspent)
CONFIG_CHANGES            0
```

```
CANARY = NOT RUN — STOPPED at the operator's own gate 8
```

## Step 2 — the exact SHA was deployed

```
run      https://github.com/joshua1602-svg/trakt/actions/runs/34750670520
head_sha 88cf5fde759c29f1cff29b9adbf003a2d6ebd530
result   success at 10:01:46Z, first attempt, no SCM-restart retry needed
```

**A ref had to be created, and this is what and why.** `workflow_dispatch` accepts
a branch or a tag and **not a SHA**, so deploying an exact commit needs a ref at
it. An annotated tag was the right vehicle and pushing one failed: this
environment's git credentials answer `HTTP 403` to a tag push, four attempts with
backoff, while branch pushes succeed. So the branch `deploy/mi-api-88cf5fde` was
created at exactly that commit and the deploy dispatched from it. It carries no
content of its own — the commit is already on
`claude/wizardly-faraday-o712ib` — and it is a vehicle, not a place work lands.

Provenance does not rest on that ref being immutable. It rests on the artefact
stamp, below.

## Step 3 — the running service IS that commit

```
PROVENANCE   PASS
  the running service reports 88cf5fde759c29f1cff29b9adbf003a2d6ebd530,
  stamped from the artefact (source='artefact')

run  https://github.com/joshua1602-svg/trakt/actions/runs/34750913505
```

`deploy-mi-api.yml` writes `build_info.json` into the staged artefact carrying
`GITHUB_SHA`, and refuses to upload unless it is forty characters. `GET
/api/health` reads that stamp back out of the **running process**, and
`source='artefact'` says it came from the stamp rather than from an environment
variable or a guess. `mi_agent_api.build_info` exists for exactly this reason:
`app.version` is hand-written and was identical across every deploy this year, so
it cannot tell one build from another and is not accepted as provenance.

**The first reading was a timing artefact and is recorded rather than hidden.**
Run 1 read `00bb3e9d` — the 11 September build — 52 seconds after the deploy
action returned success. App Service keeps `SCM_DO_BUILD_DURING_DEPLOYMENT=true`,
so Oryx builds remotely *after* that return and the process restarts after the
build; reading the old commit that soon measures the restart, not the deployment.
The re-read two minutes later returned the deployed commit. This is not a retry
seeking a favourable sample: the check is a free GET, and the thing it measures
genuinely changes when the restart lands.

## Step 4 — not triggered

Provenance was established, so the STOP on unestablished provenance did not fire.

## Step 7 — the preflight, and the blocker it found

```
AUTH    FAIL   GET /mi/catalogue returned 401: the bearer did not authenticate
TOKEN   expires 2026-09-12T18:50:04+00:00
        expired = True,  54890s past expiry  (15h 15m)
        carries_a_subject = True
SPEND   model calls 0,  /mi/query calls 0
```

**`MI_BEARER` has aged out.** It is a stored Entra access token, and the one in
the repository secret expired at 18:50:04 UTC on 12 September — a little over
fifteen hours before this run. It is present (1826 characters) and well-formed;
it is simply no longer valid.

Three things were checked before reporting that as an auth failure rather than as
a harness defect of mine:

1. **The endpoint really is protected.** `mi_agent_api.app` applies
   `Depends(auth_guard)` at the `FastAPI(...)` level, so `/mi/catalogue` is
   covered app-wide. A 401 there is a real rejection, not a route that ignores
   the header.
2. **The request was shaped the way the accepted harnesses shape it** — a single
   `Authorization: Bearer` header, as `run_serving_acceptance.py` sends.
3. **The cause was diagnosed rather than guessed.** The harness reads the token's
   own `exp` claim without verifying its signature — a diagnosis, not an
   authentication decision, since the service had already decided. Only the time
   fields and whether a subject claim exists are reported; the token is never
   printed and no other claim is.

`carries_a_subject = True` matters for the step after this one: on the bearer path
`ExecutionContext.actor_id` **is** that subject, and `plan_serving_canary` refuses
the literal `unknown-principal`, so a token without one could never satisfy the
allow-list however it were configured. This one can, once it is re-minted.

**Zero-cost by construction.** Both checks are GETs against endpoints that
resolve no question, so neither reaches `interpretation_v2` and neither bought an
Opus interpretation. The ten authorised calls are untouched.

## Steps 5, 6 and 8 — stopped, and why each is stopped

### Step 5 cannot be performed from here, by this repository

`MI_AGENT_PLAN_SERVE` and `MI_AGENT_PLAN_SERVE_PRINCIPALS` are Azure App Service
**application settings**, and changing one is an ARM control-plane call. This
repository's `trakt-mi-api` automation holds `AZURE_MI_API_PUBLISH_PROFILE`,
which grants Kudu and nothing else. Checked rather than assumed: `azure/login`
and `az webapp config appsettings set` do appear in this repo — for
`trakt-blob-trigger-v2`, `trakt-ops-api` and the Streamlit dashboard, each with
its own federated credentials. **No ARM path to `trakt-mi-api` exists here.**
`deploy/trakt-mi-api/provision.sh` sets app settings, but it is a local operator
script run under an interactive `az` login, not a CI mechanism.

So the "existing authorised App Service configuration mechanism" for this app is
the Azure Portal or an operator's own `az` session — which is exactly how the
pre-deployment state was read.

### Step 6 therefore has no before/after to verify

With no write, there is nothing to verify. For when there is, the honest
verification is behavioural rather than a second config read: a canary-handled
request writes an evidence record carrying `serving.mode` and
`principal_matched`, and `serving.mode == "canary"` with `principal_matched ==
true` is proof the flag is on and the principal is allow-listed. That record is
readable over Kudu, which this repository does hold.

### Step 8's precondition is unmet, so the canary was not run

The operator's gate is explicit: run the ten **only if** the deployed SHA is
confirmed, plan serving is confirmed on, **and authentication passes**. One of
three holds. The bank authorises one attempt with `retries: 0`, so firing it now
would spend all ten on guaranteed 401s and void the single pre-registered run.

## What an operator needs to do, precisely

Three things, and the second one is the one that costs time to find the hard way.

1. **Re-mint `MI_BEARER`.** A fresh Entra access token for the same identity,
   stored in *Settings → Secrets and variables → Actions*. The canary run must
   start within the new token's lifetime — these live tens of minutes, so mint it
   immediately before dispatching rather than in advance.

2. **Set `MI_AGENT_PLAN_SERVE` to the literal string `canary`.** Not `on`, not
   `true`, not `1`. `plan_serving_canary.serve_mode()` returns the canary mode
   only for an exact case-folded match on `canary`, and reads **anything else as
   off** — including the word the instruction used. A flag set to `on` would look
   enabled in the Portal and serve nobody, and every canary question would come
   back as a legacy answer that scores as a miss.

3. **Set `MI_AGENT_PLAN_SERVE_PRINCIPALS` to the `oid` the re-minted token
   carries.** Comma-separated, exact, case-insensitive, and wildcards are
   rejected by name: `*`, `all`, `any`, `everyone`, plus `unknown-principal` and
   `local-dev`, which name no individual. An empty or unset list serves nobody
   even with the flag on, because `handles()` requires **both**. Its current
   contents are not known from here and should be confirmed rather than assumed —
   the two earlier Slice 1B acceptance runs both ended in failure, so there is no
   successful run to infer a good allow-list from, even though run 2 did get
   envelopes back, which means the pairing was correct at some point on
   10 September.

Then the canary can be dispatched. It needs no further product change: the
deployed build is already the commit every offline gate passed on.

## Integrity

```
PRODUCT_CODE_CHANGED_SINCE_THE_TESTED_HEAD = NO   (asserted in CI by diffing)
serving_canary_bank_manifest  27b7d825…  unchanged, re-hashed in CI
live_change_form_gate     4e6fca59…  intact
live_change_form_gate_v2  b9a842c6…  intact
live_change_form_gate_v3  2ff59ff1…  intact
live_change_form_gate_v4  26345762…  intact
```

No question was asked, so no expectation could be written after seeing a result.
The bank is unspent, unedited and still hashes to its pre-registration. No
product code, no bank, no expectation and no semantic policy was touched in this
run, and no configuration was changed by it.
