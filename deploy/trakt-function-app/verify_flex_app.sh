#!/usr/bin/env bash
# Azure-side checks for trakt-blob-trigger-v2 that are VALID ON FLEX CONSUMPTION.
#
#   bash deploy/trakt-function-app/verify_flex_app.sh plan    <rg> <app>
#   bash deploy/trakt-function-app/verify_flex_app.sh indexed <rg> <app>
#
# WHAT WAS REMOVED, AND WHY
# The previous gates encoded the Dedicated / Elastic-Premium hosting model and
# were all invalid here:
#
#   SCM_DO_BUILD_DURING_DEPLOYMENT / ENABLE_ORYX_BUILD app settings
#       Dedicated-plan controls. On Flex Consumption "you don't need to set any
#       application settings to request a remote build. You instead pass a remote
#       build parameter when you start deployment." Unset is CORRECT on Flex, so
#       the gate failed a correctly configured app.
#
#   Kudu VFS probe of /home/site/wwwroot/.python_packages/...
#       Flex Consumption has no persistent wwwroot to read. One deploy is the
#       only supported technology and the package lives in a deployment blob
#       container.
#
#   Kudu /api/command import probe, and /api/deployments/latest
#       Not a Flex surface. For Flex, "your Git history and CI/CD process provide
#       the only way to track code deployments at a given point in time."
#
# WHAT REPLACES THEM
#   plan    — the app really is Flex Consumption, so the workflow's assumptions
#             hold; fails loudly if the plan ever changes underneath us.
#   indexed — a running host loaded the deployed package and registered the
#             trigger. Necessary, and explicitly NOT sufficient: the handler's
#             module-level closure needs only azure.functions (supplied by the
#             worker image), so indexing succeeds even with zero dependencies.
#             What proves the dependencies is that the deploy step ran with
#             remote-build: true and succeeded — an Oryx build is always
#             performed during a remote build on Flex, and a failed pip install
#             fails the deployment.
#
# VERIFY_APP_JSON / VERIFY_PLAN_JSON / VERIFY_FUNCTIONS_JSON substitute the az
# responses so the tests cover every branch without a subscription.
set -uo pipefail

MODE="${1:?usage: verify_flex_app.sh <plan|indexed> <resource-group> <app>}"
RG="${2:?resource group required}"
APP="${3:?function app name required}"

#: The trigger that must be registered after a deployment.
EXPECTED_FUNCTION="on_raw_blob_event"

#: Flex Consumption App Service plan SKU.
FLEX_SKU="FC1"

json_or_fail() {
  # json_or_fail <payload> <expected first char> <what>
  case "${1:0:1}" in
    "$2") return 0 ;;
    *) echo "FAIL: could not read $3. Azure CLI said:"
       printf '%s\n' "$1" | sed 's/^/      /'
       return 1 ;;
  esac
}

case "$MODE" in

plan)
  echo "=== hosting plan of $APP ==="
  APP_JSON="${VERIFY_APP_JSON:-$(az functionapp show -g "$RG" -n "$APP" -o json 2>&1)}"
  json_or_fail "$APP_JSON" "{" "the function app" || exit 1

  # Two independent signals, because neither is guaranteed to be present in every
  # az version: Flex sites carry a functionAppConfig block (its deployment
  # storage container is where one deploy uploads), and their App Service plan
  # SKU is FC1.
  read -r HAS_CONFIG FARM_ID <<<"$(printf '%s' "$APP_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
cfg = d.get('functionAppConfig') or (d.get('properties') or {}).get('functionAppConfig')
farm = d.get('serverFarmId') or (d.get('properties') or {}).get('serverFarmId') or ''
print('yes' if cfg else 'no', farm or '-')")"
  echo "  functionAppConfig present : $HAS_CONFIG"
  echo "  serverFarmId              : $FARM_ID"

  SKU="unknown"
  if [ "$FARM_ID" != "-" ]; then
    PLAN_JSON="${VERIFY_PLAN_JSON:-$(az appservice plan show --ids "$FARM_ID" -o json 2>&1)}"
    if json_or_fail "$PLAN_JSON" "{" "the App Service plan" >/dev/null 2>&1; then
      SKU="$(printf '%s' "$PLAN_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
sku = d.get('sku') or (d.get('properties') or {}).get('sku') or {}
print(sku.get('name') or 'unknown')")"
    fi
  fi
  echo "  plan SKU                  : $SKU"

  if [ "$HAS_CONFIG" = "yes" ] || [ "$SKU" = "$FLEX_SKU" ]; then
    echo "  OK: $APP is a Flex Consumption app, which is what this workflow targets"
    exit 0
  fi

  echo
  echo "FAIL: $APP does not look like a Flex Consumption app (no"
  echo "      functionAppConfig, plan SKU '$SKU' is not $FLEX_SKU)."
  echo "      This workflow deploys with Azure/functions-action and"
  echo "      remote-build: true, which is the FLEX control. On Dedicated or"
  echo "      Elastic Premium the equivalent is scm-do-build-during-deployment"
  echo "      plus enable-oryx-build, so the deployment would install nothing"
  echo "      and the trigger would fail on its first lazy import."
  echo "      Reconcile the workflow with the plan before deploying again."
  exit 1
  ;;

indexed)
  echo "=== functions registered on $APP after deployment ==="
  FUNCTIONS_JSON="${VERIFY_FUNCTIONS_JSON:-$(az functionapp function list -g "$RG" -n "$APP" -o json 2>&1)}"
  json_or_fail "$FUNCTIONS_JSON" "[" "the function list" || exit 1

  NAMES="$(printf '%s' "$FUNCTIONS_JSON" | python3 -c "
import json, sys
for f in json.load(sys.stdin):
    # az returns '<app>/<function>'; the bare name is what the host indexed.
    print((f.get('name') or '').rsplit('/', 1)[-1])")"

  if [ -z "$NAMES" ]; then
    echo "  (no functions returned)"
    echo
    echo "FAIL: the host registered no functions. The deployed package did not"
    echo "      load — check the deploy step's Oryx build output above. Note that"
    echo "      restarting the app cannot fix this: each Flex deployment"
    echo "      overwrites the package, so only a new deployment changes what runs."
    exit 1
  fi
  printf '  registered: %s\n' $NAMES

  if printf '%s\n' "$NAMES" | grep -Fxq "$EXPECTED_FUNCTION"; then
    echo "  OK: a running host loaded the deployed package and registered"
    echo "      '$EXPECTED_FUNCTION'."
    echo "  NOTE: this proves the package loaded, NOT that dependencies are"
    echo "        installed — the handler's module-level closure needs only"
    echo "        azure.functions, which the worker image provides. The"
    echo "        dependencies are proven by the deploy step having run with"
    echo "        remote-build: true and succeeded: an Oryx build is always"
    echo "        performed during a remote build on Flex Consumption, and a"
    echo "        failed install fails the deployment."
    exit 0
  fi

  echo
  echo "FAIL: '$EXPECTED_FUNCTION' is not registered. The deployed package is"
  echo "      not the one this repository builds, or indexing failed."
  exit 1
  ;;

*)
  echo "unknown mode '$MODE' (expected plan or indexed)"
  exit 1
  ;;
esac
