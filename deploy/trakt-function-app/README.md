# trakt-blob-trigger-v2 deployment checks

`.github/workflows/main_trakt.yml` builds `deploy.zip` and deploys it with
`az functionapp deployment source config-zip --build-remote true`. That command
reports success in two situations that both leave the OCC intake broken, and
neither used to be detected:

| Failure mode | Symptom in production | Guard |
| --- | --- | --- |
| A runtime package is missing from the zip | `ModuleNotFoundError: No module named 'operations_control'` on the first blob | `verify_package.py` (pre-upload) |
| The remote build installs nothing | `ModuleNotFoundError: No module named 'yaml'` at `operations_control/engine.py:25` on the first blob | `verify_remote_build.sh` (before and after upload) |

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

```bash
bash deploy/trakt-function-app/verify_remote_build.sh prereqs   trakt trakt-blob-trigger-v2
bash deploy/trakt-function-app/verify_remote_build.sh installed trakt trakt-blob-trigger-v2
```

* `prereqs` (before upload) — fails unless `SCM_DO_BUILD_DURING_DEPLOYMENT` or
  `ENABLE_ORYX_BUILD` is enabled (`true` or `1`) and `WEBSITE_RUN_FROM_PACKAGE`
  is unset or `0`. A read-only package mount gives Oryx nowhere to write.
  Prints the exact `az` command to fix whatever it found.
* `installed` (after upload) — discovers the SCM hostname from
  `enabledHostNames` (never constructs it; secure unique default hostnames
  include a hash) and probes Kudu VFS for each required distribution under
  `/home/site/wwwroot/.python_packages/lib/site-packages/`. That is the remote
  build target for Linux Python Function Apps — *not* `antenv/`, which is the
  App Service layout.

Behaviour tests, no Azure subscription required:

```bash
bash deploy/trakt-function-app/tests/test_verify_remote_build.sh
```

## What restarting does

Nothing, for either failure mode. A restart re-runs the same broken start-up
against the same `wwwroot`. Only a deployment that actually installs
dependencies fixes it.
