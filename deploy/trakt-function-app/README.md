# trakt-blob-trigger-v2 deployment (Flex Consumption)

**One workflow deploys this app: `.github/workflows/main_trakt-blob-trigger-v2.yml`.**
`main_trakt.yml` used to deploy it as well; it has been retired, and
`verify_flex_workflow.py workflow` now fails the build if a second deploying
workflow reappears.

## What went wrong, and what each check now guards

The production failure was `ModuleNotFoundError: No module named 'yaml'` at
`operations_control/engine.py:25`, on a trigger that was firing correctly. Two
deployment-architecture faults produced it, and no application code was involved.

**1. Two workflows deployed one app on every push to main.** One requested a
remote build (`az functionapp deployment source config-zip --build-remote true`),
the other used `Azure/functions-action@v1` without `remote-build`. They raced —
the loser reported `ERROR: Deployment was cancelled and another deployment is in
progress.` — and the local-build one won every time.

**2. `remote-build` defaults to `'false'`.** With it false the platform installs
nothing and the documented contract shifts to the package: *"Dependencies must be
installed into `./.python_packages/lib/site-packages`."* Nothing did that. The
old build steps ran `source venv/bin/activate` in one `run:` step and
`pip install -r requirements.txt` in the next — separate shells, so pip installed
into the runner's system Python — and `zip release.zip ./* -r` drops every
dotfile anyway, so `.python_packages` and `.funcignore` could never have shipped.

Nothing installed the dependencies, and nothing checked.

| Fault | Symptom on the first blob | Guard |
| --- | --- | --- |
| A second workflow deploys the app | random winner; a local-build package lands | `verify_flex_workflow.py workflow` |
| `remote-build` unset or false | `ModuleNotFoundError: No module named 'yaml'` | `verify_flex_workflow.py workflow` |
| A local dependency build in the package | a stale second copy of site-packages | `verify_flex_workflow.py package` |
| A runtime package missing from the zip | `ModuleNotFoundError: No module named 'operations_control'` | `verify_package.py` |
| A dependency not declared in `requirements.txt` | remote build installs it nowhere | `verify_package.py` |
| App moved off Flex Consumption | `remote-build` silently stops applying | `verify_flex_app.sh plan` |
| Deployed package not loaded | trigger never fires | `verify_flex_app.sh indexed` |

Both failure modes surface only on a real blob arrival, never at deployment time,
because the entrypoint's module-level import closure is stdlib plus
`azure.functions` — and `azure-functions` ships inside the Python worker image.
The host starts, the Event Grid trigger registers and fires, and every heavier
import (`yaml`, `pandas`, `openpyxl`) happens lazily further down the call path.
So "Event Grid and the trigger are working" tells you nothing about whether the
dependencies exist.

## Deployment configuration

```yaml
- uses: Azure/functions-action@v1
  with:
    app-name: 'trakt-blob-trigger-v2'
    package: 'deploy.zip'
    remote-build: true
```

Deliberately **not** set, and asserted absent:

| Input | Why |
| --- | --- |
| `slot-name` | deployment slots are not supported in Flex Consumption |
| `sku` | only needed with `publish-profile`; the action resolves it from RBAC |
| `scm-do-build-during-deployment` | Dedicated / Elastic-Premium control |
| `enable-oryx-build` | Dedicated / Elastic-Premium control |

Microsoft's guidance is explicit: for Flex Consumption *"don't set
scm-do-build-during-deployment or enable-oryx-build"* — an Oryx build is always
performed during a remote build on Flex.

## Checks

### verify_flex_workflow.py — static, no Azure

```bash
python3 deploy/trakt-function-app/verify_flex_workflow.py workflow
python3 deploy/trakt-function-app/verify_flex_workflow.py package deploy.zip
```

`workflow` finds every step that deploys `trakt-blob-trigger-v2` by **either**
mechanism — the action's `app-name` input *and* an `az functionapp deployment`
run line. Recognising both matters: the two racing workflows used different
mechanisms, so a check that understood only one would have seen a single deployer
and passed. It then requires `remote-build: true` and the four forbidden inputs
absent.

`package` asserts the archive carries no `.python_packages/`, `venv/`, `.venv/`
or `antenv/`. Remote build is now the single source of installed packages.

### verify_package.py — static, no Azure

```bash
python3 deploy/trakt-function-app/verify_package.py deploy.zip \
  --skip-imports --emit-requirements _check_reqs.txt
python -m pip install -r _check_reqs.txt
python3 deploy/trakt-function-app/verify_package.py deploy.zip
```

1. **Manifest** — every path the entrypoint can reach, *including through lazy
   imports inside handlers*, is in the archive.
2. **Requirements declaration** — every third-party module that closure imports
   is declared in the shipped `requirements.txt`. Remote build installs exactly
   what `requirements.txt` declares, so an undeclared dependency is a production
   failure. (The earlier check pip-installed a hardcoded list, making a missing
   declaration invisible.)
3. **Import** — unpack the real artefact and walk the whole lazy chain, not just
   `import function_app`, which succeeds even when `operations_control` is absent.

The `pip install` above is for CI's import probe only. Nothing installed on the
runner is shipped.

### verify_flex_app.sh — Azure-side

```bash
bash deploy/trakt-function-app/verify_flex_app.sh plan    trakt trakt-blob-trigger-v2
bash deploy/trakt-function-app/verify_flex_app.sh indexed trakt trakt-blob-trigger-v2
```

* `plan` (before deploy) — the app really is Flex Consumption, via two
  independent signals: a `functionAppConfig` block on the site, and plan SKU
  `FC1`. `remote-build` is the *Flex* control; if the app were moved to Dedicated
  or Elastic Premium the parameter would stop having any effect and the
  deployment would install nothing, silently.
* `indexed` (after deploy) — a running host loaded the deployed package and
  registered `on_raw_blob_event`. **Necessary, not sufficient**, and it says so:
  indexing succeeds with zero dependencies installed. What proves the
  dependencies is the deploy step having run with `remote-build: true` and
  succeeded — a failed install fails the deployment.

### Gates removed as invalid for Flex Consumption

| Removed | Why it was wrong here |
| --- | --- |
| `SCM_DO_BUILD_DURING_DEPLOYMENT` / `ENABLE_ORYX_BUILD` app-setting check | Dedicated-plan controls. On Flex *"you don't need to set any application settings to request a remote build."* Unset is **correct**, so the gate failed a correctly configured app — it is what turned `main_trakt.yml` red. |
| Kudu VFS probe of `/home/site/wwwroot/.python_packages/...` | Flex has no persistent `wwwroot` to read; the package lives in a deployment blob container. |
| Kudu `/api/command` in-site import probe | Not a Flex surface. |
| Kudu `/api/deployments/latest` status check | Not a Flex surface — *"your Git history and CI/CD process provide the only way to track code deployments at a given point in time."* |

### Tests — 36, no Azure subscription required

```bash
python3 deploy/trakt-function-app/tests/test_verify_flex_workflow.py   # 20
bash    deploy/trakt-function-app/tests/test_verify_flex_app.sh        # 16
```

`VERIFY_APP_JSON`, `VERIFY_PLAN_JSON` and `VERIFY_FUNCTIONS_JSON` inject the `az`
responses so every branch runs offline. One test asserts the invariant holds on
**this repository as committed** — otherwise the static gate would only ever be
checked against fixtures.

## What restarting does

Nothing. Each Flex Consumption deployment overwrites the current package, so a
restart re-runs the same start-up against the same package. Only a deployment
that actually installs dependencies fixes it.

## Sources

* [Deploy your Python apps to Azure Functions](https://learn.microsoft.com/en-us/azure/azure-functions/python-build-options)
* [Deployment technologies in Azure Functions](https://learn.microsoft.com/en-us/azure/azure-functions/functions-deployment-technologies)
* [Continuous delivery by using GitHub Actions](https://learn.microsoft.com/en-us/azure/azure-functions/functions-how-to-github-actions)
* [Recover from a bad Flex Consumption plan app deployment](https://learn.microsoft.com/en-us/azure/azure-functions/functions-rollback-deployments)
* [Azure/functions-action](https://github.com/Azure/functions-action)
