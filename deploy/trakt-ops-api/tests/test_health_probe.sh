#!/usr/bin/env bash
# Tests for the deployment gate's health probe (probe_health.sh).
#
# Section 1 is the regression. On 2026-08-01 the trakt-ops-api deployment
# reported status=4 (Success), Oryx installed 91 packages including gunicorn
# and uvicorn, oryx-manifest.toml was served over Kudu VFS — and the job still
# failed with:
#
#     probing https://<host>/health
#     ##[error]Process completed with exit code 28.
#
# Not one "attempt N" line was printed. The inline probe assigned curl's output
# to a variable under `set -e`, so the very first --max-time expiry (curl exit
# 28) terminated the step, defeating the eight retries written directly above
# it and skipping the container-log dump that would have explained anything.
# The only difference from the previous SUCCESSFUL deploy of the same service
# was two React files the package deliberately excludes.
#
# curl is stubbed, so these run offline and need no Azure access.
#
#   bash deploy/trakt-ops-api/tests/test_health_probe.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE="$HERE/../probe_health.sh"

PASS=0
FAIL=0
check() {
  local label="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then
    PASS=$((PASS+1)); printf '  PASS  %s\n' "$label"
  else
    FAIL=$((FAIL+1)); printf '  FAIL  %s\n     expected: %s\n     actual:   %s\n' \
      "$label" "$expected" "$actual"
  fi
}

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/bin"

# Writes a curl stub that replays the given exit-code:http-code pairs, one per
# invocation (the last pair repeats), honours -o by writing a body, and counts
# its calls in $WORK/calls.
stub_curl() {
  {
    echo '#!/usr/bin/env bash'
    echo 'CALLS="$(cat "$STUB_STATE/calls" 2>/dev/null || echo 0)"'
    echo 'CALLS=$((CALLS+1)); echo "$CALLS" > "$STUB_STATE/calls"'
    echo 'OUT=""; prev=""'
    echo 'for a in "$@"; do [ "$prev" = "-o" ] && OUT="$a"; prev="$a"; done'
    printf 'SCRIPT=(%s)\n' "$*"
    echo 'IDX=$((CALLS-1)); [ "$IDX" -ge "${#SCRIPT[@]}" ] && IDX=$((${#SCRIPT[@]}-1))'
    echo 'PAIR="${SCRIPT[$IDX]}"; RC="${PAIR%%:*}"; HTTP="${PAIR##*:}"'
    echo '[ -n "$OUT" ] && printf "{\"http\":\"%s\"}" "$HTTP" > "$OUT"'
    echo 'printf "%s" "$HTTP"'
    echo 'exit "$RC"'
  } > "$WORK/bin/curl"
  chmod +x "$WORK/bin/curl"
  : > "$WORK/calls"
}

calls() { cat "$WORK/calls" 2>/dev/null || echo 0; }

export STUB_STATE="$WORK"
export PATH="$WORK/bin:$PATH"

echo "=== 1. THE REGRESSION: a timing-out curl must not abort the caller ==="
# curl exit 28 (Operation timeout), no response body, http_code 000.
stub_curl "28:000"
OUT="$(
  set -euo pipefail                      # exactly the gate's shell options
  . "$PROBE"
  PROBE_ATTEMPTS=3 PROBE_GAP_SECONDS=0 HEALTH_BODY="$WORK/health.json" \
    probe_health "example.invalid" 2>/dev/null
  echo "|reached-the-line-after"          # the old inline probe never got here
)"
check "the step survives curl exit 28 and keeps running" \
  "000|reached-the-line-after" "$OUT"
check "all three attempts were made, not just the first" "3" "$(calls)"

echo
echo "=== 2. A cold start that comes good is a PASS, not a failure ==="
# Two timeouts while the container extracts and imports, then it serves.
stub_curl "28:000" "28:000" "0:200"
CODE="$(
  set -euo pipefail
  . "$PROBE"
  PROBE_ATTEMPTS=8 PROBE_GAP_SECONDS=0 HEALTH_BODY="$WORK/health.json" \
    probe_health "example.invalid" 2>/dev/null
)"
check "returns 200 once the service answers" "200" "$CODE"
check "stops probing as soon as it gets 200" "3" "$(calls)"

echo
echo "=== 3. A service that answers but is unhealthy reports its real code ==="
stub_curl "0:503"
CODE="$(
  set -euo pipefail
  . "$PROBE"
  PROBE_ATTEMPTS=2 PROBE_GAP_SECONDS=0 HEALTH_BODY="$WORK/health.json" \
    probe_health "example.invalid" 2>/dev/null
)"
check "503 is reported, not masked as 000" "503" "$CODE"
check "an answering-but-unhealthy service is still retried" "2" "$(calls)"

echo
echo "=== 4. The body is written so the caller can show it on failure ==="
stub_curl "0:503"
(
  set -euo pipefail
  . "$PROBE"
  PROBE_ATTEMPTS=1 PROBE_GAP_SECONDS=0 HEALTH_BODY="$WORK/body.json" \
    probe_health "example.invalid" >/dev/null 2>&1
)
check "response body is captured" '{"http":"503"}' "$(cat "$WORK/body.json")"

echo
echo "=== 5. Every attempt is announced, so a bad run is still legible ==="
stub_curl "28:000"
ERRS="$(
  set -euo pipefail
  . "$PROBE"
  PROBE_ATTEMPTS=3 PROBE_GAP_SECONDS=0 HEALTH_BODY="$WORK/health.json" \
    probe_health "example.invalid" 2>&1 >/dev/null
)"
check "three attempts are reported on stderr" "3" \
  "$(printf '%s\n' "$ERRS" | grep -c 'attempt ')"
check "each timeout is described, not left as a bare code" "3" \
  "$(printf '%s\n' "$ERRS" | grep -c 'no response within')"

echo
echo "=== 6. The cold-start budget is not a warm-request budget ==="
# 20s per attempt is what the site could not meet: it must extract
# output.tar.zst and import pandas before it can answer at all.
DEFAULT_MAX="$( . "$PROBE"; echo "$PROBE_MAX_TIME" )"
DEFAULT_ATTEMPTS="$( . "$PROBE"; echo "$PROBE_ATTEMPTS" )"
if [ "$DEFAULT_MAX" -ge 60 ]; then
  PASS=$((PASS+1)); printf '  PASS  %s\n' "per-attempt ceiling is >= 60s (was 20s)"
else
  FAIL=$((FAIL+1)); printf '  FAIL  %s\n     got: %s\n' \
    "per-attempt ceiling is >= 60s (was 20s)" "$DEFAULT_MAX"
fi
if [ "$((DEFAULT_ATTEMPTS * DEFAULT_MAX))" -ge 300 ]; then
  PASS=$((PASS+1)); printf '  PASS  %s\n' "total probe budget is >= 5 minutes"
else
  FAIL=$((FAIL+1)); printf '  FAIL  %s\n' "total probe budget is >= 5 minutes"
fi

echo
echo "=== 7. No unguarded curl-into-variable survives in the workflow ==="
# The precise defect, stated precisely: a standalone assignment adopts its
# command substitution's exit status, so `VAR="$(curl ...)"` under `set -e`
# dies the moment curl does. An inline probe is fine — an UNGUARDED one is not.
# Checks whole substitutions, since they span continuation lines.
#
# Only the workflow is scanned, and deliberately so. deploy_async.sh and
# collect_diagnostics.sh both run under `set -uo pipefail` with no `-e`, which
# is why their unguarded curl substitutions are correct: the poller has to
# survive a transient failure and the diagnostics must never abort part-way.
# `set -e` is what makes the pattern lethal, and the workflow steps are where
# it is in force.
WF="$HERE/../../../.github/workflows/deploy-ops-api.yml"
UNGUARDED="$(awk '
  /="\$\(curl/ { inblk=1; blk=""; start=NR }
  inblk        { blk = blk $0 "\n" }
  inblk && /\)"/ {
    if (blk !~ /\|\| true/) printf "  line %d: %s", start, blk
    inblk=0
  }' "$WF" 2>/dev/null || true)"
if [ -z "$UNGUARDED" ]; then
  PASS=$((PASS+1)); printf '  PASS  %s\n' "every curl-into-variable absorbs curl's exit status"
else
  FAIL=$((FAIL+1)); printf '  FAIL  %s\n%s\n' \
    "every curl-into-variable absorbs curl's exit status" "$UNGUARDED"
fi
if grep -q 'probe_health' "$WF" 2>/dev/null; then
  PASS=$((PASS+1)); printf '  PASS  %s\n' "the workflow uses the tested probe"
else
  FAIL=$((FAIL+1)); printf '  FAIL  %s\n' "the workflow uses the tested probe"
fi

echo
echo "-------------------------------------------------------------------"
printf 'passed: %d   failed: %d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
echo "All health-probe tests passed."
