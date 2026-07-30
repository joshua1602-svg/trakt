#!/usr/bin/env bash
# Verify that the Function App actually INSTALLS its dependencies.
#
#   bash deploy/trakt-function-app/verify_remote_build.sh prereqs <rg> <app>
#   bash deploy/trakt-function-app/verify_remote_build.sh installed <rg> <app>
#
# WHY THIS EXISTS
# The Functions host can start, register the Event Grid trigger and invoke the
# handler with ZERO packages installed from requirements.txt. Traced: the
# module-level import closure of the deployed root function_app.py is
# stdlib + `azure.functions` only, and azure-functions ships inside the Python
# worker image. Every heavier import — yaml, pandas — happens lazily, deeper in
# the call path.
#
# So "Event Grid and the trigger are working" is fully consistent with "no
# dependency install happened at all", and the first symptom is a
# ModuleNotFoundError on a real blob:
#
#   File "/home/site/wwwroot/operations_control/engine.py", line 25, in <module>
#     import yaml
#   ModuleNotFoundError: No module named 'yaml'
#
# `az functionapp deployment source config-zip --build-remote true` reports
# success whether or not Oryx installed anything, and nothing downstream checked.
# These two gates close that.
#
# On Linux Python Function Apps a remote build installs into
#   /home/site/wwwroot/.python_packages/lib/site-packages/
# (NOT antenv/ — that is the App Service layout).
set -uo pipefail

MODE="${1:?usage: verify_remote_build.sh <prereqs|installed> <resource-group> <app>}"
RG="${2:?resource group required}"
APP="${3:?function app name required}"

#: Distributions whose absence breaks the OCC intake, and the directory each
#: leaves in site-packages.
declare -a NEEDED_DIRS=(yaml pandas numpy openpyxl)

setting_of() {
  printf '%s' "$SETTINGS_JSON" | python3 -c "
import json,sys
d={s['name']:s.get('value') for s in json.load(sys.stdin)}
print(d.get(sys.argv[1], '<unset>'))" "$1"
}

is_enabled() {
  case "$1" in true|True|TRUE|1) return 0 ;; *) return 1 ;; esac
}

case "$MODE" in
prereqs)
  echo "=== remote-build prerequisites on $APP ==="
  # VERIFY_SETTINGS_JSON lets the tests exercise the whole gate — including the
  # `1` vs `true` handling that a previous version of this guard got wrong on the
  # Ops API — without an Azure subscription. Unset in CI, so the real path runs.
  SETTINGS_JSON="${VERIFY_SETTINGS_JSON:-$(az functionapp config appsettings list -g "$RG" -n "$APP" -o json 2>&1)}"
  if [ "${SETTINGS_JSON:0:1}" != "[" ]; then
    echo "FAIL: could not read app settings. Azure CLI said:"
    printf '%s\n' "$SETTINGS_JSON"
    exit 1
  fi

  SCM_BUILD="$(setting_of SCM_DO_BUILD_DURING_DEPLOYMENT)"
  ORYX_BUILD="$(setting_of ENABLE_ORYX_BUILD)"
  RUN_FROM_PKG="$(setting_of WEBSITE_RUN_FROM_PACKAGE)"
  echo "  SCM_DO_BUILD_DURING_DEPLOYMENT = $SCM_BUILD"
  echo "  ENABLE_ORYX_BUILD              = $ORYX_BUILD"
  echo "  WEBSITE_RUN_FROM_PACKAGE       = $RUN_FROM_PKG"

  FAILED=0
  if ! is_enabled "$SCM_BUILD" && ! is_enabled "$ORYX_BUILD"; then
    echo
    echo "FAIL: neither SCM_DO_BUILD_DURING_DEPLOYMENT nor ENABLE_ORYX_BUILD is"
    echo "      enabled (accepted: true or 1). --build-remote alone does not make"
    echo "      Oryx install anything, and the deployment still reports success —"
    echo "      the trigger then fires and dies on the first lazy import. Set:"
    echo
    echo "        az functionapp config appsettings set -g $RG -n $APP --settings \\"
    echo "          SCM_DO_BUILD_DURING_DEPLOYMENT=true ENABLE_ORYX_BUILD=true"
    FAILED=1
  fi

  # A read-only package mount leaves Oryx nowhere to write site-packages.
  if [ "$RUN_FROM_PKG" != "<unset>" ] && [ "$RUN_FROM_PKG" != "0" ]; then
    echo
    echo "FAIL: WEBSITE_RUN_FROM_PACKAGE=$RUN_FROM_PKG mounts the package"
    echo "      read-only, so a remote build cannot install into it. Remove it:"
    echo
    echo "        az functionapp config appsettings delete -g $RG -n $APP \\"
    echo "          --setting-names WEBSITE_RUN_FROM_PACKAGE"
    FAILED=1
  fi

  [ "$FAILED" -eq 0 ] && echo "  OK: a remote build will install dependencies"
  exit "$FAILED"
  ;;

installed)
  echo "=== dependencies actually present on $APP ==="
  SCM_HOST="$(az functionapp show -g "$RG" -n "$APP" \
               --query "enabledHostNames[?contains(@, '.scm.')]|[0]" -o tsv 2>/dev/null | tr -d '\r')"
  case "${SCM_HOST:-}" in
    ""|None) echo "FAIL: could not discover the SCM hostname for $APP."
             echo "      Inspect: az functionapp show -g $RG -n $APP --query enabledHostNames -o json"
             exit 1 ;;
    *.azurewebsites.net) ;;
    *) echo "FAIL: implausible SCM hostname '$SCM_HOST'"; exit 1 ;;
  esac
  echo "  SCM host: $SCM_HOST"

  TOKEN="$(az account get-access-token --resource https://management.azure.com/ \
            --query accessToken -o tsv 2>/dev/null)"
  [ -n "${TOKEN:-}" ] || { echo "FAIL: could not acquire an access token"; exit 1; }

  # Remote build target for Linux Python Functions.
  BASE="/api/vfs/site/wwwroot/.python_packages/lib/site-packages"
  probe() {
    local code
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 60 \
             -H "Authorization: Bearer $TOKEN" \
             "https://${SCM_HOST}${BASE}/$1/" 2>/dev/null)"
    printf '%s' "${code:-000}"
  }

  ROOT="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 60 \
           -H "Authorization: Bearer $TOKEN" \
           "https://${SCM_HOST}${BASE}/" 2>/dev/null)"
  echo "  site-packages root -> HTTP ${ROOT:-000}"

  MISSING=()
  for dep in "${NEEDED_DIRS[@]}"; do
    CODE="$(probe "$dep")"
    echo "  $dep -> HTTP $CODE"
    [ "$CODE" = "200" ] || MISSING+=("$dep")
  done

  if [ "${#MISSING[@]}" -ne 0 ]; then
    echo
    echo "FAIL: these dependencies are NOT installed on the Function App:"
    printf '        - %s\n' "${MISSING[@]}"
    echo
    echo "      The deployment reported success but installed nothing usable."
    echo "      The host will still start and the Event Grid trigger will still"
    echo "      fire — the module-level closure of function_app.py needs only"
    echo "      azure.functions, which the worker image provides — so the failure"
    echo "      appears on the first real blob as, for example:"
    echo "        ModuleNotFoundError: No module named 'yaml'"
    echo "        at /home/site/wwwroot/operations_control/engine.py"
    echo
    echo "      Check the remote-build prerequisites (see the 'prereqs' mode of"
    echo "      this script) and redeploy. Restarting will not install anything."
    exit 1
  fi
  echo "  OK: every dependency the OCC intake imports is installed"
  exit 0
  ;;

*)
  echo "unknown mode '$MODE' (expected prereqs or installed)"
  exit 1
  ;;
esac
