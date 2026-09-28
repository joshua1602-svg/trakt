# N01 — the HTTP 500, diagnosed

    read     run 34763214744, read-only over the Kudu VFS
    window   2026-09-13T14:23:20Z - 14:24:10Z (N01 asked 14:23:26, returned
             after 33.5s)
    spend    model calls 0, /mi/query calls 0, configuration unchanged

    VERDICT  INFRASTRUCTURE — an App Service container-restart race.
             NOT a product defect and NOT an interpretation failure.

## The platform log, inside N01's own window

Thirteen lines matched, and every one of them is a platform line. There is no
Python traceback, no application error and no exception of any kind:

    14:23:26.735  Site startup probe succeeded after 50.9167791 seconds.
    14:23:26.982  State: Starting, Action: WarmUpProbeSucceeded
    14:23:27.427  State: Starting, Action: StartingAuthContainer
    14:23:40.517  Site is running with deployment version: 57de5713-…
    14:23:40.518  Site started.
    14:23:40.526  State: Started … Site started at 09/13/2026 14:23:40 (UTC)
    14:23:40.526  Site is running with patch version PYTHON-3.11.15
    14:24:00.881  State: Stopping … LastError: SiteStartupCancelled …
                  Site started at 09/13/2026 13:59:27 (UTC)
    14:24:00.881  State: Stopping, Action: StoppingSiteContainers
    14:24:00.881  Container is terminating. Grace period: 5 seconds.
    14:24:00.931  Stop and delete container. Retry count = 0
    14:24:00.931  Stopping container: 16f50b79f61e_trakt-mi-api.
    14:24:06.390  Deleting container: 16f50b79f61e_trakt-mi-api.

**N01 was asked at 14:23:26 — the same second a new container's startup probe
succeeded, thirteen seconds before the site reported Started, and while the auth
container was still starting.** The old container, running since 13:59:27, was
torn down at 14:24:00. The request landed in the middle of a container swap.

## Why the app never saw the request, proved independently of the log

`plan_serving_canary.serve()` writes an evidence record for EVERY request that
reaches it. The tail that writes it is deliberately guarded — "RECORDING CANNOT
COST THE ANSWER" — and it runs for an orchestration error, an interpreter
failure, a clarify and a refuse alike, each with its own disposition.

The sink was polled for 120 seconds during the run and **no record for N01 was
ever written**. So the request did not reach `plan_serving_canary` at all. That
rules out, on its own and without reference to the log:

  * an interpreter or provider failure — those record `INTERPRETER_FAILURE`;
  * an orchestration error — that records `ORCHESTRATION_ERROR`;
  * a compiler clarify or refuse — those record `CLARIFY` / `REFUSE`;
  * a governed refusal of any kind — every one of them records.

Two independent lines of evidence, the platform log and the absence of a record,
agree: the failure was below the application.

## What restarted the container

Changing an Azure App Service **application setting restarts the app**. The
setting changed immediately before this run was
`MI_AGENT_PLAN_SERVE = off -> canary` — Phase 3, which the sequence requires
between provenance and the questions. The timeline fits it exactly:

    14:00:16   provenance + auth PASS       site running (started 13:59:27)
    ~14:22:35  operator sets the flag       App Service begins a restart
    14:23:09   the five are dispatched
    14:23:26   N01 asked                    startup probe succeeds this second
    14:23:40   Site started
    14:24:00   the 13:59:27 container is terminated
    14:24+     N02-N05 asked                all four served, all mode=canary

So the very act the sequence requires before the run is what made the first
question land on a process that was not yet serving. Nothing about the change was
wrong; the run simply followed it too closely.

## What this does and does not change

**It does not change the verdict.** The bank is SPENT and N01 remains a FAIL:
expectations are not relaxed after seeing results, and a case that returned no
answer did not satisfy its pinned contract.

**It does change the classification.** N01 is `INFRASTRUCTURE`, not
`SERVER_ERROR` in the product sense and not an interpretation defect. The earlier
adjudication called it "a server error … it needs a separate read-only
diagnostic"; that diagnostic has now run and the answer is a container restart.

**What remains genuinely untested.** N01's case — a comparison-shaped metric
delta on a BALANCE measure — was never interpreted, so it is untested rather than
failed. The M01 repair itself is proved live by N03, which emitted `compare`,
canonicalised to `movement`, and executed through `requested_metric` on
`run_period_change_analysis`. What N01 would have added is a second measure
family, not the repair's first proof.

## Recommendation, not performed here

Any future run should not ask its first question until the service has reported
itself ready twice in succession. The harness already reads `/` — the app's own
liveness probe, which touches no data and costs nothing — for provenance, so a
readiness gate is a few free GETs and would have prevented this entirely. It is
named rather than built: this task was asked to diagnose.

If the operator wants N01's case covered live, it needs a fresh single-question
bank; this one is spent and is not re-run.
