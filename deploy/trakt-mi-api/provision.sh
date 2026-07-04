#!/usr/bin/env bash
# Provision + deploy the Trakt MI Agent API as its own Azure App Service
# (Linux, Python) named trakt-mi-api. Idempotent-ish: re-running updates settings.
#
# Prereqs: az CLI logged in (az login), run from the REPO ROOT.
#
# Required env (export before running):
#   RESOURCE_GROUP        e.g. trakt-rg
#   APP_NAME              e.g. trakt-mi-api
#   APP_PLAN              e.g. trakt-mi-plan
#   LOCATION              e.g. uksouth
#   TRAKT_BLOB_CONNECTION storage connection string (same account as processed-v2)
# Optional:
#   PLAN_SKU              default B1
#   CLIENT_ID             default ERE (the platform/{client}/latest path)
#   CORS_ORIGINS          default * (set to your React host in prod)
set -euo pipefail

: "${RESOURCE_GROUP:?set RESOURCE_GROUP}"
: "${APP_NAME:=trakt-mi-api}"
: "${APP_PLAN:=trakt-mi-plan}"
: "${LOCATION:?set LOCATION}"
: "${TRAKT_BLOB_CONNECTION:?set TRAKT_BLOB_CONNECTION}"
PLAN_SKU="${PLAN_SKU:-B1}"
CLIENT_ID="${CLIENT_ID:-ERE}"
CORS_ORIGINS="${CORS_ORIGINS:-*}"
PLATFORM_URI="blob://processed-v2/platform/${CLIENT_ID}/latest/platform_canonical_typed.csv"
# Multi-period discovery roots (funded evolution/forecast/compare + weekly
# pipeline funnel/conversion). These are the FOLDERS that contain the dated
# cuts — funded evolution needs >=2 dated platform canonicals under the
# platform root; weekly pipeline needs the dated snapshots under the pipeline
# root. Without these the forecast/evolution/compare/funnel routes report
# "no reporting periods" / "£0" even though point-in-time works.
ONBOARDING_ROOT="blob://processed-v2/platform/${CLIENT_ID}"
PIPELINE_ROOT="blob://processed-v2/pipeline/${CLIENT_ID}"

echo ">> Resource group"
az group create -n "$RESOURCE_GROUP" -l "$LOCATION" -o none

echo ">> Linux App Service plan ($PLAN_SKU)"
az appservice plan create -g "$RESOURCE_GROUP" -n "$APP_PLAN" \
  --is-linux --sku "$PLAN_SKU" -o none

echo ">> Web App (Python 3.11)"
az webapp create -g "$RESOURCE_GROUP" -p "$APP_PLAN" -n "$APP_NAME" \
  --runtime "PYTHON:3.11" -o none

echo ">> Build during deployment + startup command"
az webapp config set -g "$RESOURCE_GROUP" -n "$APP_NAME" \
  --startup-file "bash startup.sh" -o none

echo ">> App settings (storage backend + platform canonical locator)"
az webapp config appsettings set -g "$RESOURCE_GROUP" -n "$APP_NAME" --settings \
  SCM_DO_BUILD_DURING_DEPLOYMENT=true \
  WEBSITES_PORT=8000 \
  TRAKT_STORAGE_BACKEND=blob \
  TRAKT_PROCESSED_CONTAINER=processed-v2 \
  TRAKT_BLOB_CONNECTION="$TRAKT_BLOB_CONNECTION" \
  MI_AGENT_PLATFORM_URI="$PLATFORM_URI" \
  MI_AGENT_ONBOARDING_OUTPUT_ROOT="$ONBOARDING_ROOT" \
  MI_AGENT_PIPELINE_ROOT="$PIPELINE_ROOT" \
  MI_AGENT_SCRATCH=/tmp/trakt/mi_platform \
  MI_AGENT_CORS_ORIGINS="$CORS_ORIGINS" \
  MI_API_WORKERS=2 MI_API_TIMEOUT=120 \
  -o none

echo ">> Deploy code from the repo root (Oryx builds requirements.txt)"
# Run from the REPO ROOT so the whole package tree (mi_agent_api, mi_agent,
# apps/blob_trigger_app, engine, config) ships. az webapp up zips cwd + deploys.
az webapp up -g "$RESOURCE_GROUP" -n "$APP_NAME" --plan "$APP_PLAN" \
  --runtime "PYTHON:3.11" --os-type Linux

echo ">> Restart to apply startup command + settings"
az webapp restart -g "$RESOURCE_GROUP" -n "$APP_NAME" -o none

echo ">> Done. Health: https://${APP_NAME}.azurewebsites.net/health"
