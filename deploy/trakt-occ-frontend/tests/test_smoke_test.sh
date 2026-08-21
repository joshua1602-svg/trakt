#!/usr/bin/env bash
# Tests for smoke_test.sh, driven against a local fixture frontend + API.
#
# A smoke test that cannot fail is decoration. These runs prove it PASSES a
# correctly wired deployment and FAILS each of the misconfigurations that are
# actually likely here — above all a missing CORS origin, which is the reason a
# healthy API and a healthy frontend still produce a blank dashboard.
#
#   bash deploy/trakt-occ-frontend/tests/test_smoke_test.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE="$HERE/../smoke_test.sh"
FIXTURE="$HERE/fixture_server.py"
FE_PORT=8731
API_PORT=8732
FRONTEND="http://127.0.0.1:${FE_PORT}"
API="http://127.0.0.1:${API_PORT}"
TOKEN="fixture-operator-token"
WORK="$(mktemp -d)"
SERVER_PID=""

cleanup() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  rm -rf "$WORK"
}
trap cleanup EXIT

PASS=0
FAIL=0

start_fixture() {
  [ -n "$SERVER_PID" ] && { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
  python3 "$FIXTURE" "$FE_PORT" "$API_PORT" "$API" "$@" >"$WORK/server.log" 2>&1 &
  SERVER_PID=$!
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    if curl -sS --noproxy '*' --max-time 2 -o /dev/null "$FRONTEND/" 2>/dev/null; then return 0; fi
    sleep 0.3
  done
  echo "fixture server did not start"; cat "$WORK/server.log"; exit 1
}

# The smoke test uses plain curl; make sure it does not try the sandbox proxy.
run_smoke() {
  # The health check retries, sized for an API restarting after a deployment.
  # These tests are about BRANCHING, not timing, so the retry is collapsed —
  # otherwise every unreachable-API scenario would wait over a minute.
  NO_PROXY='*' no_proxy='*' \
  HEALTH_ATTEMPTS="${HEALTH_ATTEMPTS:-2}" HEALTH_GAP_SECONDS="${HEALTH_GAP_SECONDS:-1}" \
  OPS_SMOKE_OPERATOR_TOKEN="${SMOKE_TOKEN-$TOKEN}" \
    bash "$SMOKE" "$FRONTEND" "$API" >"$WORK/out" 2>&1
  echo $?
}

# run_smoke against an API that is not listening at all.
run_smoke_no_api() {
  NO_PROXY='*' no_proxy='*' \
  HEALTH_ATTEMPTS=2 HEALTH_GAP_SECONDS=1 \
  OPS_SMOKE_OPERATOR_TOKEN="${SMOKE_TOKEN-$TOKEN}" \
    bash "$SMOKE" "$FRONTEND" "http://127.0.0.1:1" >"$WORK/out" 2>&1
  echo $?
}

expect_pass() {
  local label="$1" rc="$2"
  if [ "$rc" = "0" ]; then
    PASS=$((PASS+1)); printf '  PASS  %s\n' "$label"
  else
    FAIL=$((FAIL+1)); printf '  FAIL  %s (exit %s)\n' "$label" "$rc"
    sed 's/^/       | /' "$WORK/out"
  fi
}

expect_fail() {
  local label="$1" rc="$2" text="$3"
  if [ "$rc" != "0" ] && grep -qi "$text" "$WORK/out"; then
    PASS=$((PASS+1)); printf '  PASS  %s\n' "$label"
  else
    FAIL=$((FAIL+1))
    printf '  FAIL  %s (exit %s, expected failure mentioning "%s")\n' "$label" "$rc" "$text"
    sed 's/^/       | /' "$WORK/out"
  fi
}

echo "=== 1. A correctly wired deployment passes ==="
start_fixture
expect_pass "all checks pass against a correct fixture" "$(run_smoke)"
grep -q "signed in as: 'Administrator'" "$WORK/out" \
  && { PASS=$((PASS+1)); echo "  PASS  the authenticated principal is reported"; } \
  || { FAIL=$((FAIL+1)); echo "  FAIL  the authenticated principal was not reported"; }

echo "=== 2. Missing CORS origin is caught (the most likely real failure) ==="
start_fixture --cors-none
expect_fail "no allow-origin header -> failure naming TRAKT_OPS_CORS_ORIGINS" \
  "$(run_smoke)" "TRAKT_OPS_CORS_ORIGINS"

echo "=== 3. Wildcard CORS is rejected ==="
start_fixture --cors-wildcard
expect_fail "wildcard allow-origin -> failure" "$(run_smoke)" "CORS is wildcard"

echo "=== 4. A CORS policy that blocks the auth header is caught ==="
start_fixture --no-cors-token-header
expect_fail "allow-headers without X-Operator-Token -> failure" \
  "$(run_smoke)" "X-Operator-Token header"

echo "=== 5. A bundle built without the API URL is caught ==="
start_fixture --omit-api-url
expect_fail "missing inlined API base URL -> failure naming VITE_OPS_API_URL" \
  "$(run_smoke)" "VITE_OPS_API_URL"

echo "=== 6. A token embedded in the public bundle is caught ==="
start_fixture --embed-token
expect_fail "operator token in the bundle -> failure demanding rotation" \
  "$(run_smoke)" "rotate it"

echo "=== 6b. The UNBUILT source page is caught, not passed as healthy ==="
# This is what a real deployment served: the project directory was uploaded
# instead of dist/, so index.html referenced /src/main.tsx. It still has
# <div id="root">, so a root-element check alone reports a healthy shell.
start_fixture --unbuilt-index
RC="$(run_smoke)"
expect_fail "unbuilt index.html -> failure naming the source page" \
  "$RC" "UNBUILT source index.html"
grep -q "the project directory was uploaded instead of dist/" "$WORK/out" \
  && { PASS=$((PASS+1)); echo "  PASS  the failure names the wrong upload directory"; } \
  || { FAIL=$((FAIL+1)); echo "  FAIL  the failure did not explain the cause"; }

echo "=== 7. A missing smoke token is reported, not silently skipped ==="
start_fixture
SMOKE_TOKEN="" expect_fail "no OPS_SMOKE_OPERATOR_TOKEN -> failure naming the secret" \
  "$(SMOKE_TOKEN="" run_smoke)" "OPS_SMOKE_OPERATOR_TOKEN"

echo "=== 8. The unauthenticated boundary is asserted, not assumed ==="
start_fixture
run_smoke >/dev/null
grep -q "GET /ops/me without a token -> 401" "$WORK/out" \
  && { PASS=$((PASS+1)); echo "  PASS  unauthenticated /ops/me is verified to be 401"; } \
  || { FAIL=$((FAIL+1)); echo "  FAIL  the unauthenticated check did not run"; }


echo "=== 9. An API that is not answering is diagnosed as ONE thing ==="
# The failure that started this: the API was mid-restart, every call returned a
# connection failure, and the run reported six failures — including CORS advice
# for a service that was not answering at all. One accurate failure beats five
# confident ones.
expect_fail "an unreachable API -> a failure naming the API, not CORS" \
  "$(run_smoke_no_api)" "did not answer at all"
if grep -q "5-8. Skipped" "$WORK/out"; then
  PASS=$((PASS+1)); printf '  PASS  %s\n' "checks 5-8 are skipped rather than guessed at"
else
  FAIL=$((FAIL+1)); printf '  FAIL  %s\n' "checks 5-8 should be skipped when the API is unreachable"
fi
if grep -qi "TRAKT_OPS_CORS_ORIGINS" "$WORK/out"; then
  FAIL=$((FAIL+1)); printf '  FAIL  %s\n' "an unreachable API must not be blamed on CORS"
else
  PASS=$((PASS+1)); printf '  PASS  %s\n' "CORS is not blamed for an unreachable API"
fi
if grep -q "The frontend itself passed every check" "$WORK/out"; then
  PASS=$((PASS+1)); printf '  PASS  %s\n' "the frontend result is still reported"
else
  FAIL=$((FAIL+1)); printf '  FAIL  %s\n' "the frontend result should still be reported"
fi

echo
echo "-------------------------------------------------------------------"
printf 'passed: %d   failed: %d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
echo "All smoke-test behaviour tests passed."
