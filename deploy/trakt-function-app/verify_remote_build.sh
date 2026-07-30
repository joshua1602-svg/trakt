#!/usr/bin/env bash
# Prove that a trakt-blob-trigger-v2 deployment produced a WORKING site, not
# just a successful-looking upload.
#
#   bash deploy/trakt-function-app/verify_remote_build.sh prereqs   <rg> <app>
#   bash deploy/trakt-function-app/verify_remote_build.sh deployed  <rg> <app>
#   bash deploy/trakt-function-app/verify_remote_build.sh installed <rg> <app>
#   bash deploy/trakt-function-app/verify_remote_build.sh imports   <rg> <app>
#
# WHY THIS EXISTS
# The Functions host can start, register the Event Grid trigger and invoke the
# handler with ZERO packages installed from requirements.txt. Traced: the
# module-level import closure of the deployed root function_app.py is
# stdlib + `azure.functions`, and azure-functions ships inside the Python worker
# image. Every heavier import — yaml, pandas — happens lazily, deeper in the
# call path:
#
#   function_app.py:107   from apps.blob_trigger_app import occ_intake   (lazy)
#   occ_intake.py:29-31   from operations_control.engine import OpsEngine (lazy)
#   engine.py:25          import yaml
#
# So "Event Grid and the trigger are working" is fully consistent with "no
# dependency install happened at all", and the first symptom is a
# ModuleNotFoundError on a real blob.
#
# `az functionapp deployment source config-zip --build-remote true` reports
# success whether or not Oryx installed anything, and nothing downstream checked.
# These four gates close that, in the order the failure can occur.
#
# On Linux Python Function Apps a remote build installs into
#   /home/site/wwwroot/.python_packages/lib/site-packages/
# (NOT antenv/ — that is the App Service layout).
set -uo pipefail

MODE="${1:?usage: verify_remote_build.sh <prereqs|deployed|installed|imports> <resource-group> <app>}"
RG="${2:?resource group required}"
APP="${3:?function app name required}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE="$HERE/kudu_probe.py"

is_enabled() {
  case "$1" in true|True|TRUE|1) return 0 ;; *) return 1 ;; esac
}

# --------------------------------------------------------------------------
# Shared Kudu access for the three post-deployment modes.
# --------------------------------------------------------------------------
kudu_connect() {
  # Discover the SCM hostname; never construct it. Azure secure unique default
  # hostnames are <app>-<hash>.scm.<region>.azurewebsites.net, and building the
  # name from the app name produced `curl: (6) Could not resolve host` on the
  # Ops API.
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
}

#: kudu_get <path> <output-file> -> echoes the HTTP status code.
#: `-w '%{http_code}'` already prints 000 on a transport failure; do not append a
#: `|| echo 000` fallback, which is how a doubled `HTTP 000000` was produced.
kudu_get() {
  curl -sS -o "$2" -w '%{http_code}' --max-time 120 \
       -H "Authorization: Bearer $TOKEN" \
       "https://${SCM_HOST}$1" 2>/dev/null
}

case "$MODE" in

# --------------------------------------------------------------------------
# 1. Will a remote build install anything at all?
# --------------------------------------------------------------------------
prereqs)
  echo "=== 1. remote-build prerequisites on $APP ==="
  # VERIFY_SETTINGS_JSON lets the tests exercise the whole gate — including the
  # `1` vs `true` handling that a previous version of this guard got wrong on the
  # Ops API — without an Azure subscription. Unset in CI, so the real path runs.
  SETTINGS_JSON="${VERIFY_SETTINGS_JSON:-$(az functionapp config appsettings list -g "$RG" -n "$APP" -o json 2>&1)}"
  if [ "${SETTINGS_JSON:0:1}" != "[" ]; then
    echo "FAIL: could not read app settings. Azure CLI said:"
    printf '%s\n' "$SETTINGS_JSON"
    exit 1
  fi

  setting_of() {
    printf '%s' "$SETTINGS_JSON" | python3 -c "
import json,sys
d={s['name']:s.get('value') for s in json.load(sys.stdin)}
print(d.get(sys.argv[1], '<unset>'))" "$1"
  }

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

# --------------------------------------------------------------------------
# 2. Did the deployment itself complete successfully, per Kudu?
# --------------------------------------------------------------------------
deployed)
  echo "=== 2. Kudu's verdict on the latest deployment of $APP ==="
  # The Kudu DeployStatus enum lives in ONE place, pinned by the Ops API tests.
  # It is a Kudu-wide contract, not a per-service detail, and restating it is
  # exactly how an off-by-one (4=Failed, 5=Success) once reported a successful
  # deployment as a hard failure.
  # shellcheck source=../trakt-ops-api/kudu_status.sh
  . "$HERE/../trakt-ops-api/kudu_status.sh"

  kudu_connect
  BODY="$(mktemp)"; trap 'rm -f "$BODY"' EXIT
  CODE="$(kudu_get "/api/deployments/latest" "$BODY")"
  echo "  GET /api/deployments/latest -> HTTP $CODE"
  if [ "$CODE" != "200" ]; then
    echo "FAIL: could not read the deployment record. Response body:"
    sed 's/^/      /' "$BODY"
    exit 1
  fi

  read -r DEP_ID DEP_STATUS DEP_COMPLETE < <(python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print(d.get('id','<none>'), d.get('status'), d.get('complete'))" "$BODY")
  MESSAGE="$(python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print((d.get('message') or '').strip().replace(chr(10),' '))" "$BODY")"

  echo "  deployment id : $DEP_ID"
  echo "  status        : $DEP_STATUS ($(deploy_status_label "$DEP_STATUS"))"
  echo "  complete      : $DEP_COMPLETE"
  echo "  message       : $MESSAGE"
  python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print(f\"  errors        : {d.get('errorCount')}   warnings: {d.get('warningCount')}\")
print(f\"  deployer      : {d.get('deployer')}\")" "$BODY"

  VERDICT="$(classify_deploy_status "$DEP_STATUS" "$DEP_COMPLETE")"
  if [ "$VERDICT" = "success" ]; then
    echo "  OK: Kudu reports the deployment succeeded"
    exit 0
  fi

  echo
  echo "FAIL: Kudu classifies this deployment as '$VERDICT', not success."
  if [ "$DEP_ID" != "<none>" ]; then
    echo "      Deployment log:"
    LOGS="$(mktemp)"
    LCODE="$(kudu_get "/api/deployments/$DEP_ID/log" "$LOGS")"
    if [ "$LCODE" = "200" ]; then
      python3 -c "
import json,sys
for e in json.load(open(sys.argv[1])):
    print('      ', e.get('log_time',''), (e.get('message') or '').strip())" "$LOGS" \
        2>/dev/null || sed 's/^/      /' "$LOGS"
    else
      echo "      (could not read the log: HTTP $LCODE)"
    fi
    rm -f "$LOGS"
  fi
  exit 1
  ;;

# --------------------------------------------------------------------------
# 3. Are the dependencies actually on the site, where the worker looks?
# --------------------------------------------------------------------------
installed)
  echo "=== 3. dependencies present on $APP ==="
  kudu_connect

  # The path fix: prove the OCC code itself is there before blaming packages.
  # The reported traceback named this exact file, which is how we knew the
  # missing-package theory did not explain that error.
  ENGINE="$(mktemp)"
  ECODE="$(kudu_get "/api/vfs/site/wwwroot/operations_control/engine.py" "$ENGINE")"
  rm -f "$ENGINE"
  echo "  wwwroot/operations_control/engine.py -> HTTP $ECODE"
  if [ "$ECODE" != "200" ]; then
    echo "FAIL: the OCC engine module is not on the deployed site. The package is"
    echo "      incomplete — see deploy/trakt-function-app/verify_package.py."
    exit 1
  fi

  # Remote build target for Linux Python Functions.
  BASE="/api/vfs/site/wwwroot/.python_packages/lib/site-packages"
  LISTING="$(mktemp)"; trap 'rm -f "$LISTING"' EXIT
  CODE="$(kudu_get "$BASE/" "$LISTING")"
  echo "  GET $BASE/ -> HTTP $CODE"
  if [ "$CODE" = "404" ]; then
    echo
    echo "FAIL: site-packages does not exist at all. The remote build installed"
    echo "      NOTHING, which is exactly the state that produces"
    echo "      ModuleNotFoundError on the first blob while the trigger appears"
    echo "      healthy. Re-check the 'prereqs' gate and redeploy; restarting the"
    echo "      app cannot install anything."
    exit 1
  fi
  if [ "$CODE" != "200" ]; then
    echo "FAIL: could not list site-packages. Response body:"
    sed 's/^/      /' "$LISTING"
    exit 1
  fi

  python3 "$PROBE" site-packages --listing "$LISTING"
  exit $?
  ;;

# --------------------------------------------------------------------------
# 4. Does the chain that failed in production import on the deployed site?
# --------------------------------------------------------------------------
imports)
  echo "=== 4. the failing import chain, executed on the deployed site ==="
  # WHAT THIS DOES AND DOES NOT PROVE
  # It runs the real import chain against the real /home/site/wwwroot with the
  # worker's site-packages on PYTHONPATH — so it exercises the exact statement
  # that failed (engine.py:25 `import yaml`) and validates the installed binary
  # wheels for pandas/numpy, which a directory listing cannot.
  #
  # It is NOT an Event Grid invocation. A synthetic invocation cannot be made
  # side-effect-free: occ_intake.handle_arrival calls engine.create_batch(...)
  # BEFORE it downloads anything, so a probe event would write a real OCC input
  # batch and audit entries. A deployment gate must not mutate governance state,
  # so the import chain is exercised directly instead.
  #
  # function_app.py is deliberately NOT imported here: it needs the Azure worker
  # bindings, so a failure there would say nothing about dependencies.
  #
  # VERIFY_LINUXFX / VERIFY_DUMP_REMOTE let the tests build and syntax-check the
  # script that will run on the site without an Azure subscription. A remote
  # script cannot be checked once it is inside a JSON payload, and a shell or
  # Python typo there would surface as an opaque non-zero exit from Kudu.
  RUNTIME="${VERIFY_LINUXFX:-$(az functionapp config show -g "$RG" -n "$APP" \
              --query linuxFxVersion -o tsv 2>/dev/null | tr -d '\r')}"
  echo "  linuxFxVersion: ${RUNTIME:-<unknown>}"
  PYVER="$(printf '%s' "${RUNTIME:-}" | sed -n 's#^[Pp]ython|\([0-9][0-9.]*\)$#\1#p')"
  echo "  target python : ${PYVER:-<unknown, will fall back to python3>}"

  # Build the remote command. The SCM container is not the worker container, so
  # pick the interpreter matching the site's configured runtime where possible —
  # binary wheels installed for 3.11 must be imported by 3.11 to mean anything.
  # The version actually used is printed, so a mismatch is visible rather than
  # silently changing what the result means.
  REMOTE_SH=$(cat <<'REMOTE'
set -u
SP=/home/site/wwwroot/.python_packages/lib/site-packages
PY=""
if [ -n "${WANT_PY:-}" ]; then
  for cand in /opt/python/${WANT_PY}*/bin/python3 /usr/bin/python${WANT_PY}; do
    [ -x "$cand" ] && PY="$cand" && break
  done
fi
[ -n "$PY" ] || PY="$(command -v python3 || true)"
[ -n "$PY" ] || { echo "no python interpreter in the SCM container" >&2; exit 127; }
echo "interpreter: $PY"
"$PY" -V
PYTHONPATH="$SP:/home/site/wwwroot" "$PY" - <<'PY'
import sys
print("sys.path[0:3] =", sys.path[0:3])
import yaml, pandas, numpy, openpyxl
print("yaml", yaml.__version__, "| pandas", pandas.__version__,
      "| numpy", numpy.__version__, "| openpyxl", openpyxl.__version__)
# The exact chain the Event Grid handler walks lazily on a real blob.
from operations_control.engine import OpsEngine
from operations_control.stores import OpsStore
from apps.blob_trigger_app import occ_intake
assert hasattr(occ_intake, "handle_arrival"), "occ_intake.handle_arrival missing"
print("OCC intake import chain OK:", OpsEngine.__name__, OpsStore.__name__)
PY
REMOTE
)
  # WANT_PY is substituted here rather than inside the quoted heredoc so the
  # remote script stays literal (no accidental local expansion).
  REMOTE_SH="WANT_PY='${PYVER:-}'
$REMOTE_SH"

  if is_enabled "${VERIFY_DUMP_REMOTE:-0}"; then
    printf '%s\n' "$REMOTE_SH"
    exit 0
  fi

  kudu_connect
  REQ="$(mktemp)"; RESP="$(mktemp)"
  trap 'rm -f "$REQ" "$RESP"' EXIT
  python3 -c "
import json, sys
print(json.dumps({'command': sys.stdin.read(), 'dir': '/home/site/wwwroot'}))
" <<<"$REMOTE_SH" > "$REQ"

  CODE="$(curl -sS -o "$RESP" -w '%{http_code}' --max-time 300 \
            -X POST "https://${SCM_HOST}/api/command" \
            -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" \
            --data-binary "@$REQ" 2>/dev/null)"
  echo "  POST /api/command -> HTTP $CODE"

  if [ "$CODE" != "200" ]; then
    echo
    echo "COULD NOT RUN THE IN-SITE IMPORT CHECK (HTTP $CODE). Response body:"
    sed 's/^/      /' "$RESP"
    echo
    echo "      Kudu's command endpoint is not usable on this app — on some"
    echo "      Function App plans it is absent, and basic-auth publishing being"
    echo "      disabled can also produce 401/403 here. That is a limitation of"
    echo "      the diagnostic, not evidence about the deployment: gate 3 already"
    echo "      proved the packages are installed where the worker looks."
    echo
    echo "      Re-run with OCC_IMPORTS_PROBE_OPTIONAL=1 to downgrade this gate"
    echo "      to a warning if the endpoint is unavailable on this plan."
    if is_enabled "${OCC_IMPORTS_PROBE_OPTIONAL:-0}"; then
      echo "      OCC_IMPORTS_PROBE_OPTIONAL is set — reporting as a WARNING."
      exit 0
    fi
    exit 1
  fi

  python3 "$PROBE" command-result --json "$RESP"
  exit $?
  ;;

*)
  echo "unknown mode '$MODE' (expected prereqs, deployed, installed or imports)"
  exit 1
  ;;
esac
