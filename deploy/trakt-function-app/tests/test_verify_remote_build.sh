#!/usr/bin/env bash
# Behaviour tests for deploy/trakt-function-app/verify_remote_build.sh.
#
# The `prereqs` gate is driven entirely by app settings, so VERIFY_SETTINGS_JSON
# lets every branch run here with no Azure subscription. This matters: the
# equivalent guard on the Ops API silently accepted a misconfigured site because
# it compared against the string "true" only, while the live setting was `1`.
# These cases pin that behaviour down.
set -uo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/verify_remote_build.sh"
PASS=0
FAIL=0

settings() {
  # settings NAME=VALUE ... -> the JSON shape `az ... appsettings list -o json` emits
  local out="[" first=1 pair
  for pair in "$@"; do
    [ "$first" -eq 1 ] || out+=","
    first=0
    out+="{\"name\":\"${pair%%=*}\",\"value\":\"${pair#*=}\"}"
  done
  printf '%s]' "$out"
}

# expect <label> <expected-exit> <settings-json> [substring that must appear]
expect() {
  local label="$1" want="$2" json="$3" needle="${4:-}"
  local output rc
  output="$(VERIFY_SETTINGS_JSON="$json" bash "$SCRIPT" prereqs rg app 2>&1)"
  rc=$?
  if [ "$rc" -ne "$want" ]; then
    echo "FAIL: $label — exit $rc, expected $want"
    printf '%s\n' "$output" | sed 's/^/      | /'
    FAIL=$((FAIL + 1))
    return
  fi
  # `--` because needles legitimately start with `--` (az flags in the remedies).
  if [ -n "$needle" ] && ! printf '%s' "$output" | grep -qF -- "$needle"; then
    echo "FAIL: $label — output did not mention '$needle'"
    printf '%s\n' "$output" | sed 's/^/      | /'
    FAIL=$((FAIL + 1))
    return
  fi
  echo "pass: $label"
  PASS=$((PASS + 1))
}

echo "--- prereqs gate ---"

expect "both build flags true -> pass" 0 \
  "$(settings SCM_DO_BUILD_DURING_DEPLOYMENT=true ENABLE_ORYX_BUILD=true)" \
  "OK: a remote build will install dependencies"

# The exact live shape of the Ops API, which the earlier string comparison
# mis-read. `1` is a valid enabled value in Azure.
expect "SCM build set to 1 -> pass" 0 \
  "$(settings SCM_DO_BUILD_DURING_DEPLOYMENT=1)" \
  "OK: a remote build will install dependencies"

expect "ENABLE_ORYX_BUILD alone -> pass" 0 \
  "$(settings ENABLE_ORYX_BUILD=True)" \
  "OK:"

# This is the configuration that produces the reported production failure:
# the deployment succeeds, the trigger fires, and the first lazy import dies.
expect "neither flag set -> fail with the az remedy" 1 \
  "$(settings FUNCTIONS_WORKER_RUNTIME=python)" \
  "SCM_DO_BUILD_DURING_DEPLOYMENT=true ENABLE_ORYX_BUILD=true"

expect "flags explicitly false -> fail" 1 \
  "$(settings SCM_DO_BUILD_DURING_DEPLOYMENT=false ENABLE_ORYX_BUILD=false)" \
  "FAIL: neither SCM_DO_BUILD_DURING_DEPLOYMENT nor ENABLE_ORYX_BUILD"

expect "run-from-package blocks the build -> fail" 1 \
  "$(settings SCM_DO_BUILD_DURING_DEPLOYMENT=true WEBSITE_RUN_FROM_PACKAGE=1)" \
  "WEBSITE_RUN_FROM_PACKAGE=1 mounts the package"

expect "WEBSITE_RUN_FROM_PACKAGE=0 is not a blocker" 0 \
  "$(settings ENABLE_ORYX_BUILD=true WEBSITE_RUN_FROM_PACKAGE=0)" \
  "OK:"

expect "both problems are reported together" 1 \
  "$(settings WEBSITE_RUN_FROM_PACKAGE=https://blob/pkg.zip)" \
  "--setting-names WEBSITE_RUN_FROM_PACKAGE"

expect "unreadable az output -> fail loudly, not silently pass" 1 \
  "ERROR: (ResourceNotFound) app not found" \
  "could not read app settings"

echo "--- reported settings are echoed for the record ---"
expect "values are printed" 0 \
  "$(settings SCM_DO_BUILD_DURING_DEPLOYMENT=1 ENABLE_ORYX_BUILD=true)" \
  "SCM_DO_BUILD_DURING_DEPLOYMENT = 1"

echo "--- the script that runs ON the deployed site ---"
# It is sent to Kudu inside a JSON payload, where a shell or Python typo would
# come back only as an opaque non-zero exit code. Check it here instead.
REMOTE="$(VERIFY_LINUXFX='Python|3.11' VERIFY_DUMP_REMOTE=1 \
          bash "$SCRIPT" imports rg app 2>&1)"

check() {
  if eval "$2" >/dev/null 2>&1; then
    echo "pass: $1"; PASS=$((PASS + 1))
  else
    echo "FAIL: $1"; FAIL=$((FAIL + 1))
  fi
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
# Strip this script's own progress lines; the remote script starts at WANT_PY=.
printf '%s\n' "$REMOTE" | sed -n '/^WANT_PY=/,$p' > "$TMP/remote.sh"

check "the remote shell script is syntactically valid" \
      'bash -n "$TMP/remote.sh"'
check "it targets the interpreter matching linuxFxVersion" \
      'grep -q "WANT_PY=.3\.11." "$TMP/remote.sh"'
check "it puts the worker site-packages on PYTHONPATH" \
      'grep -q "\.python_packages/lib/site-packages" "$TMP/remote.sh"'
check "it does NOT import function_app (needs the Azure worker bindings)" \
      '! grep -q "import function_app" "$TMP/remote.sh"'

# The embedded Python must compile too — extract the heredoc body.
sed -n "/<<'PY'/,/^PY$/p" "$TMP/remote.sh" | sed '1d;$d' > "$TMP/remote.py"
check "the embedded Python is non-empty" '[ -s "$TMP/remote.py" ]'
check "the embedded Python compiles" \
      'python3 -m py_compile "$TMP/remote.py"'
check "it imports the exact chain that failed in production" \
      'grep -q "^import yaml, pandas, numpy, openpyxl$" "$TMP/remote.py" &&
       grep -q "from operations_control.engine import OpsEngine" "$TMP/remote.py" &&
       grep -q "from apps.blob_trigger_app import occ_intake" "$TMP/remote.py"'

# The payload is assembled with json.dumps, so quoting cannot break it. Prove the
# round trip rather than assuming it.
check "the remote script survives JSON encoding intact" \
      'python3 -c "
import json, pathlib, sys
src = pathlib.Path(sys.argv[1]).read_text()
assert json.loads(json.dumps({\"command\": src}))[\"command\"] == src
" "$TMP/remote.sh"'

echo "--- usage ---"
if bash "$SCRIPT" nonsense rg app >/dev/null 2>&1; then
  echo "FAIL: an unknown mode should exit non-zero"
  FAIL=$((FAIL + 1))
else
  echo "pass: unknown mode exits non-zero"
  PASS=$((PASS + 1))
fi

if bash "$SCRIPT" prereqs >/dev/null 2>&1; then
  echo "FAIL: a missing resource group should exit non-zero"
  FAIL=$((FAIL + 1))
else
  echo "pass: missing arguments exit non-zero"
  PASS=$((PASS + 1))
fi

echo
echo "passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]
