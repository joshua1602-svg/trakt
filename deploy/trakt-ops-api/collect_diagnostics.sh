#!/usr/bin/env bash
# Collect every piece of deployment evidence for trakt-ops-api after a failure.
#
# Purpose: turn "HTTP_504 GatewayTimeout" — which says nothing about cause — into
# a record that distinguishes packaging / Oryx build / startup / timeout / Kudu
# failure. Runs unconditionally on failure and never itself fails the job, so a
# missing log source cannot hide the sources that ARE available.
#
# Usage (from the repo root):
#   bash deploy/trakt-ops-api/collect_diagnostics.sh <resource-group> <app-name>
set -uo pipefail

RG="${1:?usage: collect_diagnostics.sh <resource-group> <app-name>}"
APP="${2:?app name required}"
OUT="${DIAG_DIR:-_diagnostics}"
mkdir -p "$OUT"

# Hostnames come from the site resource, never from the app name — see
# azure_hosts.sh. A diagnostics script that guesses the wrong hostname reports
# "unreachable" for a perfectly healthy site, which is worse than no output.
# shellcheck source=deploy/trakt-ops-api/azure_hosts.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/azure_hosts.sh"

# Discovery is best-effort here: if it fails, the ARM-based sections must still
# run, so record the failure and carry on rather than exiting.
SCM_URL=""
if SCM_HOST="$(discover_scm_host "$RG" "$APP" 2>"$OUT/scm_discovery_error")"; then
  SCM_URL="https://${SCM_HOST}"
else
  SCM_HOST=""
fi

APP_URL=""
if APP_HOST="$(discover_app_host "$RG" "$APP" 2>"$OUT/app_discovery_error")"; then
  APP_URL="https://${APP_HOST}"
else
  APP_HOST=""
fi

section() {
  echo
  echo "==================================================================="
  echo "$*"
  echo "==================================================================="
}

# Every collector is best-effort: one unavailable source must not stop the rest.
try() {
  local label="$1"; shift
  if "$@" 2>&1; then
    return 0
  fi
  echo "  (could not collect: $label)"
}

TOKEN="$(az account get-access-token --resource https://management.azure.com/ \
         --query accessToken -o tsv 2>/dev/null || true)"

kudu() {
  # $1 = path under the SCM site
  if [ -z "${SCM_URL:-}" ]; then
    echo "  (SCM hostname could not be discovered — see section 0; skipping Kudu)"
    return 0
  fi
  if [ -z "${TOKEN:-}" ]; then
    echo "  (no access token — cannot reach Kudu)"
    return 0
  fi
  local code
  # `-w '%{http_code}'` already prints 000 when the connection or DNS fails, so do
  # NOT append a fallback with `|| echo 000` — that produced "HTTP 000000" and made
  # the status unreadable. Capture what curl reports and default only if empty.
  code="$(curl -sS -H "Authorization: Bearer $TOKEN" --max-time 90 \
          -o "$OUT/kudu_body" -w '%{http_code}' "${SCM_URL}$1" 2>/dev/null)"
  code="${code:-000}"
  echo "  GET $1 -> HTTP $code"
  if [ -s "$OUT/kudu_body" ]; then
    # Pretty-print JSON when it is JSON; otherwise emit raw text.
    python3 -c "
import json,sys
raw = open(sys.argv[1], 'rb').read().decode('utf-8', 'replace')
try:
    print(json.dumps(json.loads(raw), indent=2)[:200000])
except Exception:
    print(raw[:200000])
" "$OUT/kudu_body" | sed 's/^/  /'
  fi
  if [ "$code" = "401" ] || [ "$code" = "403" ]; then
    echo "  NOTE: Kudu refused the token. If SCM basic auth is disabled and the"
    echo "        deploying identity lacks rights on the site, Kudu endpoints are"
    echo "        unreachable — use the ARM sources below instead."
  fi
}

section "0. DISCOVERED ENDPOINTS"
# Printed first so every later "unreachable" line can be judged against the
# hostname actually used. Neither hostname is constructed from the app name.
echo "  app name            : $APP"
echo "  resource group      : $RG"
echo "  SCM (Kudu) hostname : ${SCM_HOST:-<discovery failed>}"
echo "  public hostname     : ${APP_HOST:-<discovery failed>}"
if [ -z "${SCM_HOST:-}" ] && [ -s "$OUT/scm_discovery_error" ]; then
  echo "  --- SCM discovery error ---"
  sed 's/^/  /' "$OUT/scm_discovery_error"
fi
if [ -z "${APP_HOST:-}" ] && [ -s "$OUT/app_discovery_error" ]; then
  echo "  --- public hostname discovery error ---"
  sed 's/^/  /' "$OUT/app_discovery_error"
fi
echo "  all enabled hostnames on the site:"
az webapp show -g "$RG" -n "$APP" --query "enabledHostNames" -o tsv 2>/dev/null \
  | sed 's/^/    /' || echo "    (could not read enabledHostNames)"

section "A. DEPLOYMENT STATUS (ARM) — az webapp log deployment show"
# ARM-side view. Independent of Kudu reachability.
try "az webapp log deployment show" \
  az webapp log deployment show -g "$RG" -n "$APP" -o json

section "B. DEPLOYMENT LIST (ARM) — az webapp log deployment list"
try "az webapp log deployment list" \
  az webapp log deployment list -g "$RG" -n "$APP" -o table

section "C. LATEST KUDU DEPLOYMENT RECORD — /api/deployments/latest"
# The authoritative answer to "did it actually fail, or did only my HTTP call
# time out?". Kudu DeployStatus: 0=Pending, 1=Building, 2=Deploying, 3=Failed,
# 4=Success. complete=false means STILL RUNNING.
kudu "/api/deployments/latest"

section "D. ALL KUDU DEPLOYMENTS — /api/deployments"
kudu "/api/deployments"

section "E. KUDU DEPLOYMENT LOG + NESTED DETAILS (Oryx build output)"
# The Oryx build output is NOT in the top-level log; it hangs off log entries that
# carry a details_url. Fetch both levels, or the build failure stays invisible.
if [ -n "${TOKEN:-}" ] && [ -n "${SCM_URL:-}" ]; then
  LATEST_ID="$(curl -sS -H "Authorization: Bearer $TOKEN" --max-time 60 \
    "${SCM_URL}/api/deployments/latest" 2>/dev/null \
    | python3 -c "import json,sys; print(json.load(sys.stdin).get('id',''))" 2>/dev/null || true)"
  if [ -n "${LATEST_ID:-}" ]; then
    echo "  latest deployment id: $LATEST_ID"
    kudu "/api/deployments/$LATEST_ID/log"
    echo
    echo "  --- nested log details (this is where the Oryx build log lives) ---"
    curl -sS -H "Authorization: Bearer $TOKEN" --max-time 60 \
      "${SCM_URL}/api/deployments/$LATEST_ID/log" 2>/dev/null \
      | python3 -c "
import json, sys
try: entries = json.load(sys.stdin)
except Exception: entries = []
for e in entries:
    url = e.get('details_url')
    if url:
        print(url)
" 2>/dev/null | sort -u | while read -r detail_url; do
      [ -z "$detail_url" ] && continue
      echo "  >> $detail_url"
      curl -sS -H "Authorization: Bearer $TOKEN" --max-time 90 "$detail_url" 2>/dev/null \
        | python3 -c "
import json, sys
raw = sys.stdin.read()
try:
    for e in json.loads(raw):
        print('     ' + str(e.get('message','')).rstrip())
except Exception:
    print('     ' + raw[:100000])
" 2>/dev/null
    done
  else
    echo "  (no deployment id available)"
  fi
else
  echo "  (skipped: no access token, or the SCM hostname could not be discovered)"
fi

section "F. ORYX BUILD LOG FROM THE SITE FILESYSTEM"
# Oryx writes its build log under /home/LogFiles and, for some builds, leaves
# oryx-build.log next to the site root. VFS access is read-only.
for candidate in \
  "/api/vfs/LogFiles/oryx-build.log" \
  "/api/vfs/site/wwwroot/oryx-build.log" \
  "/api/vfs/LogFiles/kudu/trace/" \
  "/api/vfs/LogFiles/"
do
  echo "  --- $candidate ---"
  kudu "$candidate"
done

section "G. CONTAINER / RUNTIME LOGS — Kudu docker log stream"
# If the build succeeded and the CONTAINER failed to start, the reason is here:
# this is where "ModuleNotFoundError", a bad startup command, or a port-binding
# timeout appears.
kudu "/api/logs/docker"

section "H. RUNTIME LOG BUNDLE — az webapp log download"
if az webapp log download -g "$RG" -n "$APP" --log-file "$OUT/runtime_logs.zip" >/dev/null 2>&1; then
  echo "  downloaded $OUT/runtime_logs.zip"
  unzip -o -q "$OUT/runtime_logs.zip" -d "$OUT/runtime_logs" 2>/dev/null || true
  echo "  --- files in the bundle ---"
  find "$OUT/runtime_logs" -type f 2>/dev/null | sed 's/^/    /' | head -50
  echo
  echo "  --- tail of every default_docker / container log ---"
  find "$OUT/runtime_logs" -type f \( -name "*docker*" -o -name "*.log" -o -name "*.txt" \) 2>/dev/null \
    | head -20 | while read -r f; do
        echo "  >>>>> $f"
        tail -120 "$f" 2>/dev/null | sed 's/^/    /'
      done
else
  echo "  (az webapp log download failed — application logging may be off.)"
  echo "  Enable with:"
  echo "    az webapp log config -g $RG -n $APP \\"
  echo "      --docker-container-logging filesystem --application-logging filesystem --level information"
fi

section "I. CURRENT SITE CONFIGURATION"
try "az webapp config show" \
  az webapp config show -g "$RG" -n "$APP" \
    --query "{linuxFxVersion:linuxFxVersion, appCommandLine:appCommandLine, alwaysOn:alwaysOn, healthCheckPath:healthCheckPath, numberOfWorkers:numberOfWorkers}" -o json

echo
echo "  --- deployment-relevant app settings (names and presence only) ---"
az webapp config appsettings list -g "$RG" -n "$APP" -o json 2>/dev/null \
  | python3 -c "
import json, sys
watch = ('SCM_DO_BUILD_DURING_DEPLOYMENT', 'ENABLE_ORYX_BUILD',
         'WEBSITE_RUN_FROM_PACKAGE', 'WEBSITES_PORT', 'PORT',
         'WEBSITES_CONTAINER_START_TIME_LIMIT', 'OPS_API_WORKERS', 'OPS_API_TIMEOUT')
secret = ('TRAKT_OPS_OPERATORS', 'TRAKT_BLOB_CONNECTION')
try: settings = {s['name']: s.get('value') for s in json.load(sys.stdin)}
except Exception: settings = {}
for name in watch:
    print(f'    {name:38} = {settings.get(name, \"<unset>\")}')
for name in secret:
    print(f'    {name:38} = ' + ('<set, withheld>' if name in settings else '<unset>'))
" 2>/dev/null || echo "    (could not read app settings)"

section "J. LIVE HEALTH PROBE"
# Uses the DISCOVERED public hostname. Constructing <app>.azurewebsites.net here
# would fail DNS on a site with secure unique default hostnames and be misread as
# "the app is down".
if [ -n "${APP_URL:-}" ]; then
  echo "  target: ${APP_URL}/health"
  CODE="$(curl -sS -o "$OUT/health_body" -w '%{http_code}' --max-time 30 \
          "${APP_URL}/health" 2>/dev/null)"
  CODE="${CODE:-000}"
  echo "  GET /health -> HTTP $CODE"
  [ -s "$OUT/health_body" ] && sed 's/^/  /' "$OUT/health_body" && echo
else
  echo "  (skipped: the public hostname could not be discovered — see section 0)"
fi

section "K. HOW TO READ THIS"
cat <<'GUIDE'
  Section C is decisive. Match it to one of these:

  Kudu DeployStatus: 0=Pending, 1=Building, 2=Deploying, 3=Failed, 4=Success.

    complete=false, status=1 (Building)      -> the build is STILL RUNNING. Your
                                                deploy call timed out; the
                                                deployment did not fail. Fix by
                                                deploying asynchronously (this
                                                repo now does) or by shrinking
                                                the Oryx build.
    complete=true,  status=4 (Success)       -> the deployment SUCCEEDED, possibly
                                                after the client gave up. Any 504
                                                was a client-side wait, not a
                                                failure.
    complete=true,  status=3 (Failed)        -> a genuine failure. Section E's
                                                nested details carry the Oryx
                                                build output and the reason.
    no deployment record at all              -> the upload never reached Kudu.
                                                Submission/auth problem, not a
                                                build problem.
    deployment succeeded but /health fails   -> a STARTUP problem, not a
                                                deployment one. Section G shows
                                                the container log with the real
                                                error.
GUIDE
echo
echo "Diagnostics collected under: $OUT"
exit 0
