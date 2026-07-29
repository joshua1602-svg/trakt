#!/usr/bin/env bash
# Provision + deploy the Operations Control Centre API as its OWN Azure App
# Service (Linux, Python 3.11) named trakt-ops-api. Idempotent-ish: re-running
# updates the startup command and settings.
#
# This script NEVER touches trakt-mi-api. It refuses to run against that name
# (see the guard below), creates its own App Service plan by default, and sets
# only its own configuration. The MI API keeps serving mi_agent_api.app:app via
# the repo-root startup.sh; the OCC application is not mounted into it.
#
# Deployment model: App Service SOURCE deployment with an Oryx build, matching
# how trakt-mi-api is actually deployed today (--startup-file + zip deploy from
# the repo root). No container image is introduced, so there is one build path
# to reason about and no duplicated runtime wiring.
#
# Prereqs: az CLI logged in (az login), run from the REPO ROOT.
#
# Required env (export before running):
#   RESOURCE_GROUP        e.g. trakt-rg
#   LOCATION              e.g. uksouth
#   TRAKT_BLOB_CONNECTION storage connection string (same account as the MI API)
#   OPS_OPERATORS         JSON token map; see TRAKT_OPS_OPERATORS below. The API
#                         is FAIL-CLOSED (503 on every route) without it.
#   OPS_CORS_ORIGINS      exact origin(s) of the deployed React OCC, comma-separated
# Optional:
#   APP_NAME              default trakt-ops-api
#   APP_PLAN              default trakt-ops-plan (its own plan; see note below)
#   PLAN_SKU              default B1
set -euo pipefail

: "${RESOURCE_GROUP:?set RESOURCE_GROUP}"
: "${LOCATION:?set LOCATION}"
: "${TRAKT_BLOB_CONNECTION:?set TRAKT_BLOB_CONNECTION}"
: "${OPS_OPERATORS:?set OPS_OPERATORS (JSON token map; the API is fail-closed without it)}"
: "${OPS_CORS_ORIGINS:?set OPS_CORS_ORIGINS (exact React origin; wildcards are not supported)}"
APP_NAME="${APP_NAME:-trakt-ops-api}"
APP_PLAN="${APP_PLAN:-trakt-ops-plan}"
PLAN_SKU="${PLAN_SKU:-B1}"
OPS_CONTAINER="${OPS_CONTAINER:-operations-control}"

# --- Guard: never operate on the MI API ------------------------------------- #
# A mistyped APP_NAME here would rewrite the MI service's startup command and
# settings, taking the customer-facing MI API down. Refuse outright.
if [ "$APP_NAME" = "trakt-mi-api" ] || [ "$APP_PLAN" = "trakt-mi-plan" ]; then
  echo "REFUSING: this script must not target the MI API (APP_NAME=$APP_NAME," >&2
  echo "          APP_PLAN=$APP_PLAN). Deploy the MI API with" >&2
  echo "          deploy/trakt-mi-api/provision.sh instead." >&2
  exit 1
fi

if [ "$OPS_CORS_ORIGINS" = "*" ]; then
  echo "REFUSING: wildcard CORS. The OCC exposes administrator configuration" >&2
  echo "          actions; list the exact React origin(s) instead." >&2
  exit 1
fi

echo ">> Resource group"
az group create -n "$RESOURCE_GROUP" -l "$LOCATION" -o none

# Its own plan by default. Sharing trakt-mi-plan would work, but then an OCC
# workflow burning CPU (onboarding, Annex 2 XML build) competes with MI queries
# on the same instance — the isolation this split exists for. Set APP_PLAN
# explicitly if you consciously want them co-tenanted.
echo ">> Linux App Service plan ($APP_PLAN, $PLAN_SKU)"
az appservice plan create -g "$RESOURCE_GROUP" -n "$APP_PLAN" \
  --is-linux --sku "$PLAN_SKU" -o none

echo ">> Web App (Python 3.11): $APP_NAME"
az webapp create -g "$RESOURCE_GROUP" -p "$APP_PLAN" -n "$APP_NAME" \
  --runtime "PYTHON:3.11" -o none

echo ">> Build during deployment + OCC startup command"
az webapp config set -g "$RESOURCE_GROUP" -n "$APP_NAME" \
  --startup-file "bash deploy/trakt-ops-api/startup-ops.sh" -o none

echo ">> Governance container ($OPS_CONTAINER)"
# The OCC writes workflows, decisions, rules, the audit chain and the versioned
# configuration packages here. Creating it up front means the first request does
# not fail on a missing container.
az storage container create --name "$OPS_CONTAINER" \
  --connection-string "$TRAKT_BLOB_CONNECTION" -o none

echo ">> App settings"
az webapp config appsettings set -g "$RESOURCE_GROUP" -n "$APP_NAME" --settings \
  SCM_DO_BUILD_DURING_DEPLOYMENT=true \
  WEBSITES_PORT=8000 \
  PORT=8000 \
  TRAKT_STORAGE_BACKEND=blob \
  TRAKT_BLOB_CONNECTION="$TRAKT_BLOB_CONNECTION" \
  TRAKT_OPS_CONTAINER="$OPS_CONTAINER" \
  TRAKT_OPS_STAGING_ROOT=/tmp/trakt/ops_staging \
  TRAKT_OPS_MEMORY_ROOT=/tmp/trakt/ops_memory \
  TRAKT_OPS_OPERATORS="$OPS_OPERATORS" \
  TRAKT_OPS_CORS_ORIGINS="$OPS_CORS_ORIGINS" \
  TRAKT_STATE_CONTAINER="${TRAKT_STATE_CONTAINER:-trakt-state}" \
  TRAKT_PROCESSED_CONTAINER="${TRAKT_PROCESSED_CONTAINER:-processed-v2}" \
  TRAKT_RAW_CONTAINER="${TRAKT_RAW_CONTAINER:-raw-v2}" \
  OPS_API_WORKERS=1 OPS_API_TIMEOUT=300 \
  -o none

echo ">> Deploy code from the repo root (Oryx builds requirements.txt)"
# Run from the REPO ROOT so the whole runtime closure ships. Proven by import
# trace, not assumption — the OCC app reaches:
#   operations_control/  the API, engine, stores, configuration packages
#   engine/              orchestrator_agent (imported) + gate_4b/gate_5 delivery
#                        scripts (executed as subprocesses with cwd=repo root)
#   apps/                blob_trigger_app storage / layout / persistence / approvals
#   trakt_core/          governed context, errors, tenancy, policy
#   config/              every governed YAML the packages are seeded from, plus
#                        the Annex 2 XSD under config/system/
#   the repo-root Annex 2 mapping workbook (.xlsx), passed to the XML builder
az webapp up -g "$RESOURCE_GROUP" -n "$APP_NAME" --plan "$APP_PLAN" \
  --runtime "PYTHON:3.11" --os-type Linux

echo ">> Restart to apply startup command + settings"
az webapp restart -g "$RESOURCE_GROUP" -n "$APP_NAME" -o none

echo ">> Done."
echo "   Health:  https://${APP_NAME}.azurewebsites.net/health"
echo "   Who am I: curl -H 'X-Operator-Token: <admin-token>' \\"
echo "               https://${APP_NAME}.azurewebsites.net/ops/me"
echo "   React:   build the OCC UI with"
echo "               VITE_OPS_API_URL=https://${APP_NAME}.azurewebsites.net"
echo "   trakt-mi-api was NOT touched by this script."
