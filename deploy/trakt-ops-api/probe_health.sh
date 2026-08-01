#!/usr/bin/env bash
# The deployment gate's health probe.
#
# WHY THIS IS A SCRIPT AND NOT FIVE LINES OF INLINE YAML
#
# The probe used to be written inline as:
#
#     CODE="$(curl -sS -o health.json -w '%{http_code}' --max-time 20 \
#              "https://${OPS_APP_HOST}/health" 2>/dev/null)"
#
# inside a `for attempt in 1 2 3 4 5 6 7 8` loop, under `set -euo pipefail`.
#
# That loop never ran twice. A standalone assignment takes the exit status of
# its command substitution, so when curl exceeded --max-time and exited 28,
# `set -e` terminated the whole step ON THE FIRST ATTEMPT — before the `echo`
# that reports the attempt, before the retry, and before the block that dumps
# the container log. The job failed with a bare "exit code 28" and no
# diagnostics at all, on a deployment that Kudu had reported status=4
# (Success). The retries existed for exactly the case that killed them.
#
# So the contract here is deliberate:
#
#   * a transport failure is a RESULT to retry, never a reason to abandon the
#     probe — curl's non-zero exit is absorbed and reported as HTTP 000;
#   * every attempt is announced as it happens, so a run that ends badly still
#     shows how the service behaved on the way there;
#   * the function only ever RETURNS a code. Whether that code is acceptable
#     is the caller's decision, so the caller's failure path — which reads the
#     container log — stays reachable.
#
# The budget is sized for a cold start, not a warm request. This site runs with
# alwaysOn=false and an Oryx build published with --compress-destination-dir,
# so the first request after a deploy waits for the runtime to extract
# output.tar.zst and then import operations_control, mi_agent, analytics_lib
# and pandas. Twenty seconds was never enough for that; the old value only ever
# passed when the container happened to be warm already.
#
#   . deploy/trakt-ops-api/probe_health.sh
#   CODE="$(probe_health "$OPS_APP_HOST")"

# Attempts, the per-attempt curl ceiling, and the gap between attempts.
# Defaults give a worst case of roughly 13 minutes and return as soon as the
# service answers 200.
: "${PROBE_ATTEMPTS:=10}"
: "${PROBE_MAX_TIME:=60}"
: "${PROBE_GAP_SECONDS:=20}"
: "${HEALTH_BODY:=health.json}"

# probe_health <host>
# Echoes the final HTTP status code on stdout ("000" when nothing answered) and
# writes the last response body to $HEALTH_BODY. Progress goes to stderr so the
# caller can capture the code cleanly. Never returns non-zero.
probe_health() {
  local host="$1"
  local code=000
  local attempt=1

  while [ "$attempt" -le "$PROBE_ATTEMPTS" ]; do
    # `|| true` is load-bearing: without it a timeout kills a `set -e` caller.
    code="$(curl -sS -o "$HEALTH_BODY" -w '%{http_code}' \
              --max-time "$PROBE_MAX_TIME" \
              "https://${host}/health" 2>/dev/null || true)"
    code="${code:-000}"

    if [ "$code" = "000" ]; then
      printf '  attempt %s/%s: no response within %ss (cold start or dead worker)\n' \
        "$attempt" "$PROBE_ATTEMPTS" "$PROBE_MAX_TIME" >&2
    else
      printf '  attempt %s/%s: HTTP %s\n' "$attempt" "$PROBE_ATTEMPTS" "$code" >&2
    fi

    if [ "$code" = "200" ]; then
      break
    fi
    if [ "$attempt" -lt "$PROBE_ATTEMPTS" ]; then
      sleep "$PROBE_GAP_SECONDS"
    fi
    attempt=$((attempt + 1))
  done

  printf '%s' "$code"
}
