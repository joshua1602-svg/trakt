#!/usr/bin/env bash
# Post-deployment smoke test for the Operations Control Centre frontend and the
# API it talks to.
#
#   bash deploy/trakt-occ-frontend/smoke_test.sh <frontend-url> <api-url>
#
# The operator token is read from the environment, never from an argument, so it
# cannot land in a process listing or a CI command echo:
#   OPS_SMOKE_OPERATOR_TOKEN   a token from the API's TRAKT_OPS_OPERATORS map
#
# Checks, in the order a browser would hit them:
#   1  frontend responds 200
#   2  the frontend's JS and CSS assets load, and carry the built API base URL
#   3  the deployed bundle leaks no operator token
#   4  API /health responds 200
#   5  unauthenticated /ops/me is refused (auth is still enforced)
#   6  authenticated /ops/me succeeds
#   7  authenticated /ops/dashboard succeeds
#   8  CORS allows the deployed frontend origin, and only it
set -uo pipefail

FRONTEND="${1:?usage: smoke_test.sh <frontend-url> <api-url>}"
API="${2:?api url required}"
FRONTEND="${FRONTEND%/}"
API="${API%/}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PASS=0
FAIL=0
ok()  { PASS=$((PASS+1)); printf '  PASS  %s\n' "$1"; }
bad() { FAIL=$((FAIL+1)); printf '  FAIL  %s\n        %s\n' "$1" "${2:-}"; }

# HTTP status for a URL, with optional extra curl args. Never use `|| echo 000`:
# curl's -w already prints 000 on a connection failure and appending a fallback
# produces an unreadable "000000".
status_of() {
  local url="$1"; shift
  local code
  code="$(curl -sS -o "$WORK/body" -w '%{http_code}' --max-time 30 "$@" "$url" 2>/dev/null)"
  printf '%s' "${code:-000}"
}

echo "frontend: $FRONTEND"
echo "api:      $API"
echo

echo "=== 1. Frontend responds ==="
CODE="$(status_of "$FRONTEND/")"
if [ "$CODE" = "200" ]; then ok "GET / -> 200"; else bad "GET / -> $CODE (expected 200)"; fi
cp "$WORK/body" "$WORK/index.html" 2>/dev/null || true

if grep -q '<div id="root"' "$WORK/index.html" 2>/dev/null; then
  ok "the served page contains the application root element"
else
  bad "the served page does not contain the application root element" \
      "the SWA may be serving a placeholder instead of the built app"
fi

# The UNBUILT source index.html also contains <div id="root">, so the check above
# passes for it. It is distinguishable by the dev-server module entry, which no
# build output ever contains. Naming this precisely matters: it means the wrong
# directory was uploaded, not that the build or the API is broken.
if grep -q '/src/main.tsx' "$WORK/index.html" 2>/dev/null; then
  bad "the served page is the UNBUILT source index.html" \
      "it references /src/main.tsx, so the project directory was uploaded instead of dist/ — the app cannot boot"
else
  ok "the served page is a build output, not the source page"
fi

echo "=== 2. Frontend assets load ==="
# Read the asset paths out of the served HTML rather than guessing hashed names.
ASSETS="$(grep -oE '/assets/[A-Za-z0-9._-]+\.(js|css)' "$WORK/index.html" 2>/dev/null | sort -u)"
if [ -z "$ASSETS" ]; then
  bad "no /assets/*.js or *.css referenced by index.html" \
      "the build output may not have been uploaded"
else
  while read -r asset; do
    [ -z "$asset" ] && continue
    CODE="$(status_of "${FRONTEND}${asset}")"
    if [ "$CODE" = "200" ]; then
      ok "asset $asset -> 200"
      cat "$WORK/body" >> "$WORK/bundle_all"
    else
      bad "asset $asset -> $CODE (expected 200)"
    fi
  done <<< "$ASSETS"
fi

# The API base URL must be baked in. If it is missing, the app will call its own
# origin and every request will 404 in the browser with no obvious cause.
if [ -s "$WORK/bundle_all" ] && grep -qF "$API" "$WORK/bundle_all"; then
  ok "the bundle was built against $API"
else
  bad "the deployed bundle does not contain the API base URL" \
      "VITE_OPS_API_URL was probably not set at build time"
fi

echo "=== 3. No operator token in the deployed bundle ==="
# The token is entered by the operator at runtime and kept in localStorage; it
# must never be compiled into a publicly served asset.
if [ -s "$WORK/bundle_all" ]; then
  if [ -n "${OPS_SMOKE_OPERATOR_TOKEN:-}" ] \
     && grep -qF "$OPS_SMOKE_OPERATOR_TOKEN" "$WORK/bundle_all"; then
    bad "the smoke-test operator token appears in the public bundle" \
        "a token has been embedded at build time — rotate it immediately"
  else
    ok "no operator token found in the public bundle"
  fi
  # 'trakt_ops_token' is the localStorage KEY name and is expected; a value is not.
  if grep -qF "trakt_ops_token" "$WORK/bundle_all"; then
    ok "the bundle references the token storage key (expected) and no value"
  fi
else
  bad "could not inspect the bundle" "no asset content was downloaded"
fi

echo "=== 4. API health ==="
CODE="$(status_of "$API/health")"
if [ "$CODE" = "200" ]; then
  ok "GET /health -> 200"
  sed 's/^/        /' "$WORK/body"; echo
  if grep -q '"service":"operations-control"' "$WORK/body"; then
    ok "the API identifies as operations-control"
  else
    bad "the API did not identify as operations-control" "wrong service at $API"
  fi
  if grep -q '"auth_configured":true' "$WORK/body"; then
    ok "operator authentication is configured"
  else
    bad "auth_configured is false" "TRAKT_OPS_OPERATORS is unset; every route will answer 503"
  fi
else
  bad "GET /health -> $CODE (expected 200)"
fi

echo "=== 5. Unauthenticated access is refused ==="
CODE="$(status_of "$API/ops/me")"
if [ "$CODE" = "401" ]; then
  ok "GET /ops/me without a token -> 401"
else
  bad "GET /ops/me without a token -> $CODE (expected 401)" \
      "the authentication boundary may have been weakened"
fi

echo "=== 6-7. Authenticated access ==="
if [ -z "${OPS_SMOKE_OPERATOR_TOKEN:-}" ]; then
  bad "cannot verify authenticated access" \
      "set the repository secret OPS_SMOKE_OPERATOR_TOKEN to a token from the API's TRAKT_OPS_OPERATORS map"
else
  CODE="$(status_of "$API/ops/me" -H "X-Operator-Token: ${OPS_SMOKE_OPERATOR_TOKEN}")"
  if [ "$CODE" = "200" ]; then
    ok "GET /ops/me with a token -> 200"
    # Report the role without echoing the token.
    python3 -c "
import json, sys
try:
    p = json.load(open(sys.argv[1]))['principal']
except Exception:
    print('        (could not parse the principal)'); raise SystemExit(0)
print(f\"        signed in as: {p.get('name')!r}  role: {p.get('role')!r}  admin: {p.get('is_admin')}\")
" "$WORK/body" 2>/dev/null
  else
    bad "GET /ops/me with a token -> $CODE (expected 200)" \
        "the token may not be present in TRAKT_OPS_OPERATORS on this API"
  fi

  CODE="$(status_of "$API/ops/dashboard" -H "X-Operator-Token: ${OPS_SMOKE_OPERATOR_TOKEN}")"
  if [ "$CODE" = "200" ]; then
    ok "GET /ops/dashboard with a token -> 200"
  else
    bad "GET /ops/dashboard with a token -> $CODE (expected 200)"
  fi
fi

echo "=== 8. CORS allows the deployed frontend origin ==="
# This is the single most likely reason a healthy API and a healthy frontend
# still produce a blank dashboard: the browser blocks the call before it is made.
ORIGIN="$(printf '%s' "$FRONTEND" | sed -E 's#^(https?://[^/]+).*#\1#')"
HDRS="$(curl -sS -D - -o /dev/null --max-time 30 -X OPTIONS "$API/ops/me" \
        -H "Origin: $ORIGIN" \
        -H "Access-Control-Request-Method: GET" \
        -H "Access-Control-Request-Headers: x-operator-token" 2>/dev/null)"
ALLOW_ORIGIN="$(printf '%s' "$HDRS" | grep -i '^access-control-allow-origin:' | tr -d '\r' || true)"
ALLOW_HEADERS="$(printf '%s' "$HDRS" | grep -i '^access-control-allow-headers:' | tr -d '\r' || true)"

if printf '%s' "$ALLOW_ORIGIN" | grep -qF "$ORIGIN"; then
  ok "CORS allows $ORIGIN"
elif printf '%s' "$ALLOW_ORIGIN" | grep -q '\*'; then
  bad "CORS is wildcard" "this API administers platform configuration; list the exact origin"
else
  bad "CORS does not allow $ORIGIN" \
      "set TRAKT_OPS_CORS_ORIGINS=$ORIGIN on the trakt-ops-api App Service"
fi

if printf '%s' "$ALLOW_HEADERS" | grep -qi 'x-operator-token'; then
  ok "CORS allows the X-Operator-Token request header"
else
  bad "CORS does not allow the X-Operator-Token header" \
      "the browser will block every authenticated call: ${ALLOW_HEADERS:-<no header returned>}"
fi

echo
echo "-------------------------------------------------------------------"
printf 'passed: %d   failed: %d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
echo "Smoke test passed."
