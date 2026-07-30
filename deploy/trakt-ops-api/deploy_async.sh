#!/usr/bin/env bash
# Asynchronous zip deployment of trakt-ops-api, with polling.
#
# WHY ASYNCHRONOUS
# A synchronous `az webapp deploy` holds one HTTP request open to the app's SCM
# (Kudu) front end for the whole build. Azure App Service's front end enforces a
# fixed ~230-second idle timeout on any inbound request and cannot be configured
# to wait longer; when the build outlives it, the caller is handed a 502/504 even
# though the build is still running server-side. That is why a large Oryx build
# reports "HTTP_504 GatewayTimeout" a few minutes in.
#   Reference: Microsoft Learn — "Azure App Service: there is a 230-second
#   timeout for requests that are not responded to" (App Service troubleshooting
#   / FAQ, and the same limit documented for Kudu ZipDeploy).
#   Reference: Kudu wiki, "Deploying from a zip file" — `isAsync=true` returns
#   202 immediately and the deployment is polled via /api/deployments/latest.
#
# So: fire the deployment with isAsync=true, then poll the deployment record. The
# HTTP request lasts seconds; the build takes as long as it takes.
#
# Usage (from the repo root):
#   bash deploy/trakt-ops-api/deploy_async.sh <zip> <resource-group> <app-name> [timeout-seconds]
#
# Requires: az CLI already logged in (this repo uses OIDC), python3, curl.
set -uo pipefail

ZIP="${1:?usage: deploy_async.sh <zip> <resource-group> <app-name> [timeout-seconds]}"
RG="${2:?resource group required}"
APP="${3:?app name required}"
DEADLINE_SECONDS="${4:-1800}"
POLL_SECONDS="${POLL_SECONDS:-15}"

# Discover the SCM hostname from the site resource. It is NEVER built from the
# app name: a site with secure unique default hostnames lives at
# <app>-<hash>.scm.<region>.azurewebsites.net, and the legacy guess
# <app>.scm.azurewebsites.net does not resolve — which appears as
# "curl: (6) Could not resolve host" and HTTP 000.
# shellcheck source=deploy/trakt-ops-api/azure_hosts.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/azure_hosts.sh"
# shellcheck source=deploy/trakt-ops-api/kudu_status.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/kudu_status.sh"

echo ">> Discovering the SCM (Kudu) endpoint from the site resource"
SCM_HOST="$(discover_scm_host "$RG" "$APP")" || exit 1
SCM_URL="https://${SCM_HOST}"
echo "   SCM hostname: $SCM_HOST"

echo ">> Acquiring an AAD token for the SCM endpoint"
# Kudu accepts an ARM-audience bearer token. This works whether or not SCM basic
# authentication is disabled on the site, so it does not depend on publishing
# credentials being enabled.
TOKEN="$(az account get-access-token --resource https://management.azure.com/ \
          --query accessToken -o tsv 2>/dev/null)"
if [ -z "${TOKEN:-}" ]; then
  echo "FAIL: could not acquire an access token. Is the Azure login step present?"
  exit 1
fi

echo ">> Starting asynchronous zip deployment"
echo "   target: ${SCM_URL}/api/zipdeploy?isAsync=true"
HTTP_BODY_FILE="$(mktemp)"
HTTP_HDR_FILE="$(mktemp)"
STATUS="$(curl -sS -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/zip" \
  --data-binary "@${ZIP}" \
  --max-time 600 \
  -D "$HTTP_HDR_FILE" \
  -o "$HTTP_BODY_FILE" \
  -w '%{http_code}' \
  "${SCM_URL}/api/zipdeploy?isAsync=true")"

echo "   HTTP $STATUS"
echo "--- response headers ---"
sed 's/^/   /' "$HTTP_HDR_FILE"
if [ -s "$HTTP_BODY_FILE" ]; then
  echo "--- response body ---"
  head -c 2000 "$HTTP_BODY_FILE" | sed 's/^/   /'
  echo
fi

# 202 Accepted is the documented async response. Anything else is a real failure
# at submission time and must not be reported as a successful start.
if [ "$STATUS" != "202" ] && [ "$STATUS" != "200" ]; then
  echo "FAIL: the SCM endpoint refused the upload (HTTP $STATUS)."
  echo "      This is a SUBMISSION failure, not a build failure."
  exit 1
fi

echo
echo ">> Polling the deployment record (no synchronous HTTP wait)"
START="$(date +%s)"
LAST_LOG_COUNT=0

while :; do
  ELAPSED=$(( $(date +%s) - START ))
  if [ "$ELAPSED" -gt "$DEADLINE_SECONDS" ]; then
    echo "FAIL: deployment did not complete within ${DEADLINE_SECONDS}s."
    echo "      The deployment record above is the authoritative state — the"
    echo "      build may still be running. Diagnostics follow."
    exit 1
  fi

  LATEST="$(curl -sS -H "Authorization: Bearer $TOKEN" --max-time 60 \
            "${SCM_URL}/api/deployments/latest" 2>/dev/null)"

  PARSED="$(printf '%s' "$LATEST" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print('UNPARSEABLE|||'); raise SystemExit(0)
# Status codes are interpreted by classify_deploy_status (kudu_status.sh), which
# owns the DeployStatus enum. 'status_text'/'progress' carry the human detail.
print('|'.join([
    str(d.get('id', '')),
    str(d.get('status', '')),
    str(d.get('complete', '')),
    str(d.get('status_text', '') or d.get('progress', '') or '').replace('|', '/'),
]))
" 2>/dev/null)"

  ID="$(printf '%s' "$PARSED" | cut -d'|' -f1)"
  STATUS_CODE="$(printf '%s' "$PARSED" | cut -d'|' -f2)"
  COMPLETE="$(printf '%s' "$PARSED" | cut -d'|' -f3)"
  DETAIL="$(printf '%s' "$PARSED" | cut -d'|' -f4)"

  printf '   [%4ds] status=%s (%s) complete=%s %s\n' \
    "$ELAPSED" "${STATUS_CODE:-?}" "$(deploy_status_label "${STATUS_CODE:-}")" \
    "${COMPLETE:-?}" "${DETAIL:-}"

  # Stream new log lines as they appear, so a long build is observable rather
  # than a silent wait.
  if [ -n "$ID" ] && [ "$ID" != "UNPARSEABLE" ]; then
    LOGS="$(curl -sS -H "Authorization: Bearer $TOKEN" --max-time 60 \
            "${SCM_URL}/api/deployments/$ID/log" 2>/dev/null)"
    COUNT="$(printf '%s' "$LOGS" | python3 -c "
import json,sys
try: print(len(json.load(sys.stdin)))
except Exception: print(0)
" 2>/dev/null)"
    if [ "${COUNT:-0}" -gt "${LAST_LOG_COUNT:-0}" ]; then
      printf '%s' "$LOGS" | python3 -c "
import json, sys
skip = int(sys.argv[1])
try: entries = json.load(sys.stdin)
except Exception: entries = []
for e in entries[skip:]:
    print('          ' + str(e.get('message', '')).rstrip())
" "$LAST_LOG_COUNT"
      LAST_LOG_COUNT="$COUNT"
    fi
  fi

  case "$(classify_deploy_status "${STATUS_CODE:-}" "${COMPLETE:-}")" in
    success)
      echo
      echo ">> Deployment SUCCEEDED after ${ELAPSED}s (status ${STATUS_CODE})."
      exit 0
      ;;
    failed)
      echo
      echo "FAIL: the deployment completed with status ${STATUS_CODE} (Failed)"
      echo "      after ${ELAPSED}s. This is a genuine BUILD/DEPLOY failure,"
      echo "      not a timeout. Diagnostics follow."
      exit 1
      ;;
    unexpected)
      echo
      echo "FAIL: the deployment reported complete=${COMPLETE} with an"
      echo "      unrecognised status '${STATUS_CODE}'. Failing closed rather"
      echo "      than assuming success. Diagnostics follow."
      exit 1
      ;;
  esac

  sleep "$POLL_SECONDS"
done
