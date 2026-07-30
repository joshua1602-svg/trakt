#!/usr/bin/env bash
# Behaviour tests for deploy/trakt-function-app/verify_flex_app.sh.
#
# Both gates are driven entirely by az JSON responses, so VERIFY_APP_JSON,
# VERIFY_PLAN_JSON and VERIFY_FUNCTIONS_JSON let every branch run with no Azure
# subscription. The fixtures use the real response shapes, including the
# `<app>/<function>` naming that `az functionapp function list` returns.
set -uo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/verify_flex_app.sh"
PASS=0
FAIL=0

# expect <label> <mode> <expected-exit> <needle> [env assignments...]
expect() {
  local label="$1" mode="$2" want="$3" needle="$4"; shift 4
  local output rc
  output="$(env "$@" bash "$SCRIPT" "$mode" rg trakt-blob-trigger-v2 2>&1)"
  rc=$?
  if [ "$rc" -ne "$want" ]; then
    echo "FAIL: $label — exit $rc, expected $want"
    printf '%s\n' "$output" | sed 's/^/      | /'
    FAIL=$((FAIL + 1)); return
  fi
  if [ -n "$needle" ] && ! printf '%s' "$output" | grep -qF -- "$needle"; then
    echo "FAIL: $label — output did not mention '$needle'"
    printf '%s\n' "$output" | sed 's/^/      | /'
    FAIL=$((FAIL + 1)); return
  fi
  echo "pass: $label"
  PASS=$((PASS + 1))
}

FARM='/subscriptions/s/resourceGroups/rg/providers/Microsoft.Web/serverfarms/flexplan'
FLEX_APP="{\"name\":\"trakt-blob-trigger-v2\",\"serverFarmId\":\"$FARM\",
           \"functionAppConfig\":{\"deployment\":{\"storage\":{\"type\":\"blobContainer\"}}}}"
DEDICATED_APP="{\"name\":\"trakt-blob-trigger-v2\",\"serverFarmId\":\"$FARM\"}"
FLEX_PLAN='{"name":"flexplan","sku":{"name":"FC1","tier":"FlexConsumption"}}'
EP_PLAN='{"name":"epplan","sku":{"name":"EP1","tier":"ElasticPremium"}}'

echo "--- plan gate ---"

expect "flex app (functionAppConfig + FC1) passes" plan 0 \
  "is a Flex Consumption app" \
  "VERIFY_APP_JSON=$FLEX_APP" "VERIFY_PLAN_JSON=$FLEX_PLAN"

expect "FC1 alone is enough" plan 0 \
  "plan SKU                  : FC1" \
  "VERIFY_APP_JSON=$DEDICATED_APP" "VERIFY_PLAN_JSON=$FLEX_PLAN"

expect "functionAppConfig alone is enough" plan 0 \
  "functionAppConfig present : yes" \
  "VERIFY_APP_JSON=$FLEX_APP" "VERIFY_PLAN_JSON=$EP_PLAN"

# The case that matters: if the app is moved off Flex, remote-build silently
# stops having any effect and the deployment installs nothing.
expect "elastic premium app fails with the reason" plan 1 \
  "the equivalent is scm-do-build-during-deployment" \
  "VERIFY_APP_JSON=$DEDICATED_APP" "VERIFY_PLAN_JSON=$EP_PLAN"

expect "nested ARM properties shape is understood" plan 0 \
  "is a Flex Consumption app" \
  "VERIFY_APP_JSON={\"properties\":{\"functionAppConfig\":{\"deployment\":{}},\"serverFarmId\":\"$FARM\"}}" \
  "VERIFY_PLAN_JSON=$FLEX_PLAN"

expect "unreadable app response fails loudly" plan 1 \
  "could not read the function app" \
  "VERIFY_APP_JSON=ERROR: (ResourceNotFound) app not found"

expect "unreadable plan response does not silently pass a non-flex app" plan 1 \
  "does not look like a Flex Consumption app" \
  "VERIFY_APP_JSON=$DEDICATED_APP" "VERIFY_PLAN_JSON=ERROR: forbidden"

echo "--- indexed gate ---"

LIVE='[{"name":"trakt-blob-trigger-v2/on_raw_blob_event","config":{}}]'

expect "the expected trigger is registered" indexed 0 \
  "OK: a running host loaded the deployed package" \
  "VERIFY_FUNCTIONS_JSON=$LIVE"

expect "it says plainly what it does NOT prove" indexed 0 \
  "NOT that dependencies are" \
  "VERIFY_FUNCTIONS_JSON=$LIVE"

expect "an empty list fails" indexed 1 \
  "the host registered no functions" \
  "VERIFY_FUNCTIONS_JSON=[]"

expect "an empty list explains that restarting cannot help" indexed 1 \
  "each Flex deployment" \
  "VERIFY_FUNCTIONS_JSON=[]"

expect "a different function registered fails" indexed 1 \
  "is not registered" \
  'VERIFY_FUNCTIONS_JSON=[{"name":"trakt-blob-trigger-v2/something_else"}]'

expect "a bare function name (no app prefix) is accepted" indexed 0 \
  "OK: a running host loaded the deployed package" \
  'VERIFY_FUNCTIONS_JSON=[{"name":"on_raw_blob_event"}]'

expect "unreadable function list fails loudly" indexed 1 \
  "could not read the function list" \
  "VERIFY_FUNCTIONS_JSON=ERROR: (BadRequest) host not running"

echo "--- usage ---"
if bash "$SCRIPT" nonsense rg app >/dev/null 2>&1; then
  echo "FAIL: an unknown mode should exit non-zero"; FAIL=$((FAIL + 1))
else
  echo "pass: unknown mode exits non-zero"; PASS=$((PASS + 1))
fi
if bash "$SCRIPT" plan >/dev/null 2>&1; then
  echo "FAIL: missing arguments should exit non-zero"; FAIL=$((FAIL + 1))
else
  echo "pass: missing arguments exit non-zero"; PASS=$((PASS + 1))
fi

echo
echo "passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]
