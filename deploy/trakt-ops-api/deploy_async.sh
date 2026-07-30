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

SCM="https://${APP}.scm.azurewebsites.net"

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
echo "   target: $SCM/api/zipdeploy?isAsync=true"
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
  "$SCM/api/zipdeploy?isAsync=true")"

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
            "$SCM/api/deployments/latest" 2>/dev/null)"

  PARSED="$(printf '%s' "$LATEST" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print('UNPARSEABLE|||'); raise SystemExit(0)
# Kudu status codes: 0/1=Pending, 2=Building, 3=Deploying, 4=Failed, 5=Success.
# 'complete' is the field to branch on; 'status_text'/'progress' carry detail.
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

  printf '   [%4ds] status=%s complete=%s %s\n' \
    "$ELAPSED" "${STATUS_CODE:-?}" "${COMPLETE:-?}" "${DETAIL:-}"

  # Stream new log lines as they appear, so a long build is observable rather
  # than a silent wait.
  if [ -n "$ID" ] && [ "$ID" != "UNPARSEABLE" ]; then
    LOGS="$(curl -sS -H "Authorization: Bearer $TOKEN" --max-time 60 \
            "$SCM/api/deployments/$ID/log" 2>/dev/null)"
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

  if [ "$COMPLETE" = "True" ] || [ "$COMPLETE" = "true" ]; then
    # 4 = Failed, 5 = Success in Kudu's DeployStatus enum.
    if [ "$STATUS_CODE" = "4" ]; then
      echo
      echo "FAIL: the deployment completed with status 4 (Failed) after ${ELAPSED}s."
      echo "      This is a genuine BUILD/DEPLOY failure, not a timeout."
      exit 1
    fi
    echo
    echo ">> Deployment completed after ${ELAPSED}s (status $STATUS_CODE)."
    exit 0
  fi

  sleep "$POLL_SECONDS"
done
