# trakt-blob-trigger-v2 deployment checks

`.github/workflows/main_trakt.yml` builds `deploy.zip` and deploys it with
`az functionapp deployment source config-zip --build-remote true`. That command
reports success in two situations that both leave the OCC intake broken, and
neither used to be detected:

| Failure mode | Symptom in production | Guard |
| --- | --- | --- |
| A runtime package is missing from the zip | `ModuleNotFoundError: No module named 'operations_control'` on the first blob | `verify_package.py` (pre-upload) |
| The remote build installs nothing | `ModuleNotFoundError: No module named 'yaml'` at `operations_control/engine.py:25` on the first blob | `verify_remote_build.sh` (four gates, before and after upload) |

Both failures surface only on a real blob arrival, never at deployment time,
because the entrypoint's module-level import closure is stdlib plus
`azure.functions` — and `azure-functions` ships inside the Python worker image.
The host starts, the Event Grid trigger registers and fires, and every heavier
import (`yaml`, `pandas`, `openpyxl`) happens lazily further down the call path.
So "Event Grid and the trigger are working" tells you nothing about whether the
dependencies exist.

## verify_package.py

```bash
python deploy/trakt-function-app/verify_package.py deploy.zip \
  --skip-imports --emit-requirements _check_reqs.txt   # no deps needed
python -m pip install -r _check_reqs.txt
python deploy/trakt-function-app/verify_package.py deploy.zip   # full run
```

1. **Manifest** — every path the entrypoint can reach, *including through lazy
   imports inside handlers*, is in the archive.
2. **Requirements declaration** — every third-party module that closure imports
   is declared in the shipped `requirements.txt`. The previous check
   pip-installed a hardcoded list, so a missing declaration was invisible: CI
   installed it by name while the Function App installed only what
   `requirements.txt` asked for.
3. **Import** — unpack the real artefact and walk the whole lazy chain, not just
   `import function_app`, which succeeds even when `operations_control` is absent.

## verify_remote_build.sh

Four gates, in the order the failure can occur. All three post-deployment modes
discover the SCM hostname from `enabledHostNames` and never construct it —
secure unique default hostnames carry a hash, and building the name from the app
name produced `curl: (6) Could not resolve host` on the Ops API.

```bash
bash deploy/trakt-function-app/verify_remote_build.sh prereqs   trakt trakt-blob-trigger-v2
bash deploy/trakt-function-app/verify_remote_build.sh deployed  trakt trakt-blob-trigger-v2
bash deploy/trakt-function-app/verify_remote_build.sh installed trakt trakt-blob-trigger-v2
bash deploy/trakt-function-app/verify_remote_build.sh imports   trakt trakt-blob-trigger-v2
```

**1. `prereqs`** — before upload. Fails unless `SCM_DO_BUILD_DURING_DEPLOYMENT`
or `ENABLE_ORYX_BUILD` is enabled (`true` or `1`) and `WEBSITE_RUN_FROM_PACKAGE`
is unset or `0`; a read-only package mount gives Oryx nowhere to write. Prints
the exact `az` command to fix whatever it found.

**2. `deployed`** — Kudu's own verdict on the latest deployment. `az ...
config-zip` exiting 0 is not the same as `DeployStatus == Success`. The enum is
*not* restated here: it is sourced from `../trakt-ops-api/kudu_status.sh`, which
24 tests pin — an off-by-one copy of it (`4=Failed, 5=Success`) once reported a
successful deployment as a hard failure. On anything but success the deployment
log is dumped.

**3. `installed`** — the artefacts are on the site. First asserts
`wwwroot/operations_control/engine.py` exists (the reported traceback named that
exact file, which is how the missing-package theory was ruled out for it), then
lists `/home/site/wwwroot/.python_packages/lib/site-packages/` — the remote
build target for Linux Python Function Apps, *not* `antenv/`, which is the App
Service layout — and requires both an importable package directory and a
`.dist-info` install record for `yaml`, `pandas`, `numpy` and `openpyxl`,
reporting the installed versions.

**4. `imports`** — the chain actually runs. Executes the real import chain
against the real `/home/site/wwwroot` with the worker's site-packages on
`PYTHONPATH`, via Kudu `/api/command`, choosing the interpreter that matches
`linuxFxVersion` (binary wheels built for 3.11 must be imported by 3.11 to mean
anything) and printing the version used so a mismatch is visible. `function_app`
is deliberately not imported — it needs the Azure worker bindings, so a failure
there would say nothing about dependencies.

### What gate 4 is not

It is not an Event Grid invocation, and it cannot be made into one safely:
`occ_intake.handle_arrival` calls `engine.create_batch(...)` **before** it
downloads anything, so a synthetic probe event — even for a blob that does not
exist — writes a real OCC input batch and audit entries. A deployment gate must
not mutate governance state, so the import chain is exercised directly instead.

If Kudu's command endpoint is unavailable on the app's plan (or basic-auth
publishing is disabled), gate 4 fails with the HTTP status and body and says so
explicitly rather than guessing. `OCC_IMPORTS_PROBE_OPTIONAL=1` downgrades *that
specific case* to a warning; a real `ModuleNotFoundError` still fails.

### Tests

36 cases, no Azure subscription required. Fixtures are injected for the app
settings (`VERIFY_SETTINGS_JSON`), the Kudu responses, and the script that runs
on the site (`VERIFY_DUMP_REMOTE=1`, which prints it instead of sending it —
useful for debugging, and how the tests syntax-check both the remote shell and
its embedded Python, since a typo inside a JSON payload would otherwise come
back only as an opaque non-zero exit code).

```bash
bash   deploy/trakt-function-app/tests/test_verify_remote_build.sh   # 20
python3 deploy/trakt-function-app/tests/test_kudu_probe.py           # 16
```

## What restarting does

Nothing, for either failure mode. A restart re-runs the same broken start-up
against the same `wwwroot`. Only a deployment that actually installs
dependencies fixes it.
