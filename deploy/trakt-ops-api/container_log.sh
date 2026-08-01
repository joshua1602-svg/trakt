#!/usr/bin/env bash
# Fetch the App Service container log — the actual text, not the index.
#
# WHY THIS EXISTS
#
# When a build succeeds and the service still does not serve, the reason is in
# the container log: ModuleNotFoundError, a bad startup command, a worker that
# never bound the port. Both places that went looking for it did this:
#
#     curl .../api/logs/docker | tail -c 20000
#
# `/api/logs/docker` returns a JSON INDEX of log files — machine name, size,
# lastUpdated, href — not their contents. So a real incident, in which the
# service answered nothing to the deployment gate, to the diagnostics probe and
# to an operator's browser, produced a "CONTAINER / RUNTIME LOGS" section
# containing two filenames and their byte counts. The log itself was never read.
#
# The index has to be followed: /home/LogFiles/<name> is served at
# /api/vfs/LogFiles/<name>.
#
#   . deploy/trakt-ops-api/container_log.sh
#   fetch_container_log "$SCM_HOST" "$TOKEN" out/container.log 200

# fetch_container_log <scm-host> <bearer-token> <save-to> [tail-lines]
# Prints the tail of the newest container log, or a plain reason it could not be
# read. Best-effort by contract: never returns non-zero, so a missing log cannot
# stop the collectors around it.
fetch_container_log() {
  local scm_host="$1" token="$2" save_to="$3" tail_lines="${4:-200}"
  local index_body path code

  if [ -z "${scm_host:-}" ]; then
    echo "(no SCM hostname — cannot read the container log)"; return 0
  fi
  if [ -z "${token:-}" ]; then
    echo "(no access token — cannot read the container log)"; return 0
  fi

  index_body="$(mktemp)"
  code="$(curl -sS -H "Authorization: Bearer $token" --max-time 60 \
           -o "$index_body" -w '%{http_code}' \
           "https://${scm_host}/api/logs/docker" 2>/dev/null || true)"
  if [ "${code:-000}" != "200" ]; then
    echo "(could not list container logs: HTTP ${code:-000})"
    rm -f "$index_body"; return 0
  fi

  # Newest by lastUpdated. A site that has moved between workers lists one file
  # per machine, and only the current one describes this deployment.
  path="$(python3 -c "
import json, sys
try:
    entries = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(0)
if not isinstance(entries, list) or not entries:
    raise SystemExit(0)
newest = max(entries, key=lambda e: str(e.get('lastUpdated') or ''))
print(str(newest.get('path') or '').lstrip('/'))
" "$index_body" 2>/dev/null || true)"
  rm -f "$index_body"

  if [ -z "${path:-}" ]; then
    echo "(the log index named no file — the container may never have started)"
    return 0
  fi

  # /home/LogFiles/x.log  ->  /api/vfs/LogFiles/x.log
  code="$(curl -sS -H "Authorization: Bearer $token" --max-time 90 \
           -o "$save_to" -w '%{http_code}' \
           "https://${scm_host}/api/vfs/${path#home/}" 2>/dev/null || true)"
  if [ "${code:-000}" != "200" ] || [ ! -s "$save_to" ]; then
    echo "(could not read /$path: HTTP ${code:-000})"
    return 0
  fi

  echo "/$path (saved to $save_to) — last $tail_lines lines:"
  tail -n "$tail_lines" "$save_to"
}
