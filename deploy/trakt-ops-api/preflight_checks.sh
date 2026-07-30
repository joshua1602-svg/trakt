#!/usr/bin/env bash
# Pre-deployment evidence for trakt-ops-api: what is in the artefact, and how the
# target App Service is actually configured RIGHT NOW.
#
# Both halves run BEFORE any upload, so a failed deployment can be attributed to
# packaging or configuration without re-running anything.
#
# Usage (from the repo root):
#   bash deploy/trakt-ops-api/preflight_checks.sh <zip-path> <resource-group> <app-name>
#
# Exits non-zero only for conditions that are KNOWN to break a zip deploy with an
# Oryx build. Everything else is reported and left to the reader.
set -uo pipefail

ZIP="${1:?usage: preflight_checks.sh <zip> <resource-group> <app-name>}"
RG="${2:?resource group required}"
APP="${3:?app name required}"

FAILED=0
note()  { printf '  %s\n' "$*"; }
bad()   { printf '  FAIL: %s\n' "$*"; FAILED=1; }
warn()  { printf '  WARN: %s\n' "$*"; }

echo "==================================================================="
echo "1. ARTEFACT CONTENTS"
echo "==================================================================="
if [ ! -f "$ZIP" ]; then
  echo "FAIL: $ZIP does not exist"
  exit 1
fi
ls -lh "$ZIP"
echo
echo "--- full listing (unzip -l) ---"
unzip -l "$ZIP"

# Materialise the entry list ONCE. Do not pipe `unzip` into `grep -q` per lookup:
# grep exits on first match, unzip dies of SIGPIPE, and under `pipefail` the
# pipeline reports failure — so a file that IS present is reported MISSING purely
# because it appears early in the listing. A diagnostic that lies is worse than
# no diagnostic, so every lookup below reads this file.
MANIFEST="$(mktemp)"
unzip -Z1 "$ZIP" > "$MANIFEST"
trap 'rm -f "$MANIFEST"' EXIT
echo "  archive entries: $(wc -l < "$MANIFEST")"

echo
echo "--- entries at the ARCHIVE ROOT (no directory separator) ---"
# Oryx looks for requirements.txt at the root of the deployed package. A zip with
# everything nested one level down (a common `zip -r x.zip myrepo/` mistake) builds
# nothing and starts nothing, so prove the root layout rather than assume it.
awk '!/\//' "$MANIFEST" | sort
echo

echo "--- required root files ---"
root_has() { grep -Fxq "$1" "$MANIFEST"; }

if root_has "requirements.txt"; then
  note "requirements.txt         PRESENT at archive root (Oryx build input)"
else
  bad  "requirements.txt is NOT at the archive root — Oryx will not install anything"
fi

if root_has ".deployment"; then
  note ".deployment             PRESENT at archive root"
else
  # Not a failure. On Linux App Service the Oryx build is driven by the
  # SCM_DO_BUILD_DURING_DEPLOYMENT app setting (checked in section 2), not by a
  # .deployment file; .deployment selects a custom deployment script or project
  # sub-path, neither of which this service uses. Reported so its absence is a
  # recorded observation rather than an unknown.
  note ".deployment             ABSENT (expected: not used by this service; the"
  note "                          Oryx build is driven by the"
  note "                          SCM_DO_BUILD_DURING_DEPLOYMENT app setting)"
fi

echo
echo "--- required paths inside the artefact ---"
for path in \
  deploy/trakt-ops-api/startup-ops.sh \
  operations_control/api/app.py \
  engine/orchestrator_agent/__init__.py \
  apps/blob_trigger_app/storage.py \
  trakt_core/errors.py \
  config/system/fields_registry.yaml
do
  if grep -Fxq "$path" "$MANIFEST"; then
    note "PRESENT  $path"
  else
    bad  "MISSING  $path"
  fi
done

echo
echo "--- top-level directories in the artefact ---"
awk -F/ 'NF>1 {print $1}' "$MANIFEST" | sort -u | sed 's/^/  /'

echo
echo "==================================================================="
echo "2. TARGET APP SERVICE CONFIGURATION (live query, before upload)"
echo "==================================================================="
echo "app: $APP    resource group: $RG"
echo

SETTINGS_JSON="$(az webapp config appsettings list -g "$RG" -n "$APP" -o json 2>&1)"
# Case-match rather than pipe into head/grep: an early-exiting pipeline stage
# SIGPIPEs its producer, and `pipefail` would turn that into a false failure.
if [ "${SETTINGS_JSON:0:1}" != "[" ]; then
  echo "FAIL: could not read app settings. Azure CLI said:"
  printf '%s\n' "$SETTINGS_JSON"
  exit 1
fi

setting_of() {
  printf '%s' "$SETTINGS_JSON" \
    | python3 -c "import json,sys; d={s['name']:s.get('value') for s in json.load(sys.stdin)}; print(d.get(sys.argv[1], '<unset>'))" "$1"
}

SCM_BUILD="$(setting_of SCM_DO_BUILD_DURING_DEPLOYMENT)"
ORYX_BUILD="$(setting_of ENABLE_ORYX_BUILD)"
RUN_FROM_PKG="$(setting_of WEBSITE_RUN_FROM_PACKAGE)"

echo "--- deployment-relevant app settings ---"
note "SCM_DO_BUILD_DURING_DEPLOYMENT = $SCM_BUILD"
note "ENABLE_ORYX_BUILD              = $ORYX_BUILD"
note "WEBSITE_RUN_FROM_PACKAGE       = $RUN_FROM_PKG"
note "WEBSITES_PORT                  = $(setting_of WEBSITES_PORT)"
note "PORT                           = $(setting_of PORT)"
note "TRAKT_OPS_OPERATORS            = $([ "$(setting_of TRAKT_OPS_OPERATORS)" = '<unset>' ] && echo '<unset>  (service will answer 503 on every route)' || echo '<set, value withheld>')"
note "TRAKT_OPS_CORS_ORIGINS         = $(setting_of TRAKT_OPS_CORS_ORIGINS)"
note "OPS_API_WORKERS                = $(setting_of OPS_API_WORKERS)"
note "OPS_API_TIMEOUT                = $(setting_of OPS_API_TIMEOUT)"

echo
echo "--- site configuration ---"
CONFIG_JSON="$(az webapp config show -g "$RG" -n "$APP" -o json 2>&1)"
printf '%s' "$CONFIG_JSON" | python3 -c "
import json, sys
c = json.load(sys.stdin)
for key in ('linuxFxVersion', 'appCommandLine', 'numberOfWorkers',
            'alwaysOn', 'healthCheckPath', 'use32BitWorkerProcess'):
    print(f'  {key:24} = {c.get(key)!r}')
" 2>/dev/null || { echo "  could not parse site config:"; printf '%s\n' "$CONFIG_JSON"; }

echo
echo "--- SCM (Kudu) basic-auth publishing state ---"
# If basic auth is disabled, Kudu polling must use an AAD bearer token. Recording
# this makes a later 401 from Kudu self-explanatory.
az resource show -g "$RG" --namespace Microsoft.Web --resource-type basicPublishingCredentialsPolicies \
   --name scm --parent "sites/$APP" --query "properties.allow" -o tsv 2>/dev/null \
   | sed 's/^/  scmBasicAuthAllowed = /' || note "scmBasicAuthAllowed = <could not read>"

echo
echo "==================================================================="
echo "3. VERDICT"
echo "==================================================================="

# Known-breaking combination: run-from-package makes /home/site/wwwroot a
# read-only mount of the uploaded zip, so an Oryx build has nowhere to write its
# virtual environment. Microsoft documents these as mutually exclusive.
if [ "$RUN_FROM_PKG" != "<unset>" ] && [ "$RUN_FROM_PKG" != "0" ]; then
  bad "WEBSITE_RUN_FROM_PACKAGE=$RUN_FROM_PKG is incompatible with a zip deploy"
  bad "that needs an Oryx build. Remove the setting, or deploy a pre-built package."
fi

# Azure accepts 1 as well as true for these flags, and the live site is currently
# set to SCM_DO_BUILD_DURING_DEPLOYMENT=1. Comparing only against "true" would
# report a correctly configured site as broken — this guard passed previously only
# because ENABLE_ORYX_BUILD happened to be "true" as well.
is_enabled() {
  case "$1" in
    true|True|TRUE|1) return 0 ;;
    *) return 1 ;;
  esac
}
if ! is_enabled "$SCM_BUILD" && ! is_enabled "$ORYX_BUILD"; then
  bad "Neither SCM_DO_BUILD_DURING_DEPLOYMENT nor ENABLE_ORYX_BUILD is enabled"
  bad "(accepted values: true or 1), so dependencies will not be installed and"
  bad "the worker will fail to import uvicorn."
fi

if [ "$FAILED" -eq 0 ]; then
  echo "  Pre-flight checks passed."
else
  echo "  Pre-flight checks FAILED — see FAIL lines above. Not uploading."
fi
exit "$FAILED"
