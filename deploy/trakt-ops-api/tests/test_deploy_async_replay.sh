#!/usr/bin/env bash
# End-to-end replay of deploy_async.sh against fixture Kudu responses.
#
# Stubs `az` and `curl` on PATH and drives the REAL script through the exact
# status sequence of the 2026-07-30T07:54Z run (0 -> 1 -> 4/complete), asserting
# it now exits 0. That run genuinely succeeded — "Deployment successful …
# Errors (0)" — but the script exited 1 because of an off-by-one status enum, so
# a unit test of the classifier alone is not enough: the wiring is what failed.
#
# Also replays a genuine failure (status 3) and an unrecognised terminal status.
#
#   bash deploy/trakt-ops-api/tests/test_deploy_async_replay.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/../deploy_async.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PASS=0
FAIL=0

SCM_FIXTURE="trakt-ops-api-bveyecbeh6ebarfa.scm.westeurope-01.azurewebsites.net"

cat > "$WORK/az" <<'STUB'
#!/usr/bin/env bash
case "$*" in
  *"contains(@, '.scm.')"*) printf '%s\n' "$FIXTURE_SCM" ;;
  *defaultHostName*)        printf '%s\n' "${FIXTURE_SCM/.scm./.}" ;;
  *get-access-token*)       printf 'FIXTURE-TOKEN\n' ;;
  *)                        printf 'None\n' ;;
esac
STUB

# Stub curl: serves the POST as 202, then walks $SEQUENCE_FILE one line per
# /deployments/latest poll. Honours -D / -o / -w the way the script uses them.
cat > "$WORK/curl" <<'STUB'
#!/usr/bin/env bash
hdr=""; out=""; url=""; want_code=0
prev=""
for a in "$@"; do
  case "$prev" in
    -D) hdr="$a" ;;
    -o) out="$a" ;;
  esac
  case "$a" in
    -w) want_code=1 ;;
    https://*) url="$a" ;;
  esac
  prev="$a"
done

case "$url" in
  *zipdeploy*)
    [ -n "$hdr" ] && printf 'HTTP/1.1 202 Accepted\r\nSCM-DEPLOYMENT-ID: fixture-id\r\n\r\n' > "$hdr"
    [ -n "$out" ] && : > "$out"
    [ "$want_code" = "1" ] && printf '202'
    exit 0
    ;;
  */api/deployments/latest)
    # Pop the next line of the sequence.
    idx_file="$SEQUENCE_FILE.idx"
    idx="$(cat "$idx_file" 2>/dev/null || echo 1)"
    total="$(wc -l < "$SEQUENCE_FILE")"
    [ "$idx" -gt "$total" ] && idx="$total"
    line="$(sed -n "${idx}p" "$SEQUENCE_FILE")"
    echo $((idx + 1)) > "$idx_file"
    printf '%s' "$line"
    exit 0
    ;;
  */log)
    printf '[]'
    exit 0
    ;;
esac
printf ''
exit 0
STUB

chmod +x "$WORK/az" "$WORK/curl"
printf 'fixture zip\n' > "$WORK/ops-api.zip"

run_replay() {
  # $1 = newline-separated JSON polls
  printf '%s\n' "$1" > "$WORK/seq"
  rm -f "$WORK/seq.idx"
  PATH="$WORK:$PATH" \
  FIXTURE_SCM="$SCM_FIXTURE" \
  SEQUENCE_FILE="$WORK/seq" \
  POLL_SECONDS=0 \
    bash "$SCRIPT" "$WORK/ops-api.zip" trakt-rg trakt-ops-api 60 > "$WORK/out" 2>&1
  echo $?
}

expect() {
  local label="$1" want_rc="$2" want_text="$3" rc="$4"
  if [ "$rc" = "$want_rc" ] && grep -q "$want_text" "$WORK/out"; then
    PASS=$((PASS+1)); printf '  PASS  %s\n' "$label"
  else
    FAIL=$((FAIL+1))
    printf '  FAIL  %s\n     expected rc=%s and text %s; got rc=%s\n' \
      "$label" "$want_rc" "$want_text" "$rc"
    sed 's/^/       | /' "$WORK/out"
  fi
}

echo "=== 1. Replay of the real 2026-07-30 run (0 -> 1 -> 4 complete) ==="
RC="$(run_replay '{"id":"ef5d9230","status":0,"complete":false,"progress":"Receiving changes."}
{"id":"ef5d9230","status":1,"complete":false,"progress":"Building and Deploying."}
{"id":"ef5d9230","status":4,"complete":true,"status_text":""}')"
expect "the successful deployment is reported as SUCCEEDED (exit 0)" \
  "0" "Deployment SUCCEEDED" "$RC"

echo "=== 2. The discovered SCM host is used, never assembled ==="
if grep -q "https://${SCM_FIXTURE}/api/zipdeploy?isAsync=true" "$WORK/out"; then
  PASS=$((PASS+1)); printf '  PASS  %s\n' "zipdeploy targeted the discovered secure-unique SCM host"
else
  FAIL=$((FAIL+1)); printf '  FAIL  %s\n' "zipdeploy did not target the discovered SCM host"
  sed 's/^/       | /' "$WORK/out"
fi

echo "=== 3. A genuine failure (status 3) is reported as a failure ==="
RC="$(run_replay '{"id":"x","status":1,"complete":false,"progress":"Building."}
{"id":"x","status":3,"complete":true,"status_text":""}')"
expect "status 3 terminal -> exit 1, named a genuine failure" \
  "1" "genuine BUILD/DEPLOY failure" "$RC"

echo "=== 4. An unrecognised terminal status fails closed ==="
RC="$(run_replay '{"id":"x","status":7,"complete":true,"status_text":""}')"
expect "status 7 terminal -> exit 1, fails closed" \
  "1" "unrecognised status" "$RC"

echo "=== 5. A build that never completes hits the deadline, not a false pass ==="
RC="$(printf '%s\n' '{"id":"x","status":1,"complete":false,"progress":"Building."}' > "$WORK/seq"
  rm -f "$WORK/seq.idx"
  PATH="$WORK:$PATH" FIXTURE_SCM="$SCM_FIXTURE" SEQUENCE_FILE="$WORK/seq" POLL_SECONDS=1 \
    bash "$SCRIPT" "$WORK/ops-api.zip" trakt-rg trakt-ops-api 2 > "$WORK/out" 2>&1
  echo $?)"
expect "never-completing build -> exit 1 with a deadline message" \
  "1" "did not complete within" "$RC"

echo
echo "-------------------------------------------------------------------"
printf 'passed: %d   failed: %d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
echo "All deploy_async replay tests passed."
