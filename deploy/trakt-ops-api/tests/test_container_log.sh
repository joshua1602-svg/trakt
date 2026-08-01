#!/usr/bin/env bash
# Tests for fetching the App Service container log (container_log.sh).
#
# The regression: when trakt-ops-api stopped answering, the diagnostics section
# headed "CONTAINER / RUNTIME LOGS" printed this and nothing else —
#
#     GET /api/logs/docker -> HTTP 200
#     [ { "machineName": "lw1sdlwk0002ZG", "size": 58003, ... } ]
#
# — because /api/logs/docker returns an INDEX of log files, not their contents.
# Both readers (the deployment gate and collect_diagnostics.sh) tailed the index
# and reported filenames where the error was meant to be, so an incident with
# three independent "no response" observations produced no explanation at all.
#
# The index fixture in section 1 is the literal response from that run.
#
# curl is stubbed, so these run offline and need no Azure access.
#
#   bash deploy/trakt-ops-api/tests/test_container_log.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB="$HERE/../container_log.sh"

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
contains() {
  local label="$1" needle="$2" hay="$3"
  case "$hay" in
    *"$needle"*) PASS=$((PASS+1)); printf '  PASS  %s\n' "$label" ;;
    *) FAIL=$((FAIL+1)); printf '  FAIL  %s\n     wanted to contain: %s\n     got: %s\n' \
         "$label" "$needle" "$hay" ;;
  esac
}

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/bin"
export STUB_STATE="$WORK"
export PATH="$WORK/bin:$PATH"

# The literal /api/logs/docker response from the 2026-08-01 run: two machines,
# and only the newer one describes the current deployment.
cat > "$WORK/index.json" <<'JSON'
[
  {
    "machineName": "lw0sdlwk0002JQ",
    "lastUpdated": "2026-07-29T22:12:59.2510264Z",
    "size": 41729,
    "path": "/home/LogFiles/2026_07_29_lw0sdlwk0002JQ_docker.log"
  },
  {
    "machineName": "lw1sdlwk0002ZG",
    "lastUpdated": "2026-08-01T19:39:06.8679684Z",
    "size": 58003,
    "path": "/home/LogFiles/2026_08_01_lw1sdlwk0002ZG_docker.log"
  }
]
JSON

cat > "$WORK/container.txt" <<'LOG'
2026-08-01T19:35:02 Starting container for site
2026-08-01T19:35:44 [INFO] Starting gunicorn 26.0.0
2026-08-01T19:38:12 ModuleNotFoundError: No module named 'analytics_lib'
2026-08-01T19:38:12 [ERROR] Worker (pid:71) exited with code 3
LOG

# A curl stub that serves the index for /api/logs/docker and the log body for
# /api/vfs/..., honours -o, and records every URL it was asked for.
write_stub() {
  local vfs_code="${1:-200}"
  cat > "$WORK/bin/curl" <<STUB
#!/usr/bin/env bash
OUT=""; URL=""; prev=""
for a in "\$@"; do
  [ "\$prev" = "-o" ] && OUT="\$a"
  case "\$a" in https://*) URL="\$a" ;; esac
  prev="\$a"
done
echo "\$URL" >> "$WORK/urls"
case "\$URL" in
  */api/logs/docker)
    [ -n "\$OUT" ] && cp "$WORK/index.json" "\$OUT"
    printf '200'; exit 0 ;;
  */api/vfs/*)
    if [ "$vfs_code" = "200" ]; then
      [ -n "\$OUT" ] && cp "$WORK/container.txt" "\$OUT"
    else
      [ -n "\$OUT" ] && : > "\$OUT"
    fi
    printf '%s' "$vfs_code"; exit 0 ;;
esac
printf '404'; exit 0
STUB
  chmod +x "$WORK/bin/curl"
  : > "$WORK/urls"
}

echo "=== 1. THE REGRESSION: the log text is fetched, not the index ==="
write_stub 200
OUT="$( . "$LIB"; fetch_container_log "scm.example" "tok" "$WORK/saved.log" 200 )"
contains "the real error reaches the output" "No module named 'analytics_lib'" "$OUT"
contains "the worker exit is shown too" "exited with code 3" "$OUT"
if printf '%s' "$OUT" | grep -q 'machineName'; then
  FAIL=$((FAIL+1)); printf '  FAIL  %s\n' "the JSON index is not printed in place of the log"
else
  PASS=$((PASS+1)); printf '  PASS  %s\n' "the JSON index is not printed in place of the log"
fi

echo
echo "=== 2. It follows the index to the right file ==="
check "the newest machine's log is the one fetched" \
  "https://scm.example/api/vfs/LogFiles/2026_08_01_lw1sdlwk0002ZG_docker.log" \
  "$(grep '/api/vfs/' "$WORK/urls")"
check "the /home prefix is translated to the VFS root" "1" \
  "$(grep -c '/api/vfs/LogFiles/' "$WORK/urls")"

echo
echo "=== 3. The saved copy is the whole log, for the artefact upload ==="
check "the file is saved in full" "4" "$(wc -l < "$WORK/saved.log")"

echo
echo "=== 4. Every failure is a stated reason, never a crash ==="
write_stub 404
OUT="$( set -euo pipefail; . "$LIB"
        fetch_container_log "scm.example" "tok" "$WORK/x.log" 50 )"
contains "an unreadable log says so" "could not read" "$OUT"

OUT="$( set -euo pipefail; . "$LIB"
        fetch_container_log "" "tok" "$WORK/x.log" 50 )"
contains "a missing SCM host says so" "no SCM hostname" "$OUT"

OUT="$( set -euo pipefail; . "$LIB"
        fetch_container_log "scm.example" "" "$WORK/x.log" 50 )"
contains "a missing token says so" "no access token" "$OUT"

echo
echo "=== 5. Both readers use it — neither tails the index any more ==="
WF="$HERE/../../../.github/workflows/deploy-ops-api.yml"
DIAG="$HERE/../collect_diagnostics.sh"
for f in "$WF" "$DIAG"; do
  name="$(basename "$f")"
  if grep -q 'fetch_container_log' "$f" 2>/dev/null; then
    PASS=$((PASS+1)); printf '  PASS  %s uses fetch_container_log\n' "$name"
  else
    FAIL=$((FAIL+1)); printf '  FAIL  %s uses fetch_container_log\n' "$name"
  fi
  if grep -qE 'logs/docker"? *[|].*tail' "$f" 2>/dev/null; then
    FAIL=$((FAIL+1)); printf '  FAIL  %s no longer tails the log index\n' "$name"
  else
    PASS=$((PASS+1)); printf '  PASS  %s no longer tails the log index\n' "$name"
  fi
done

echo
echo "-------------------------------------------------------------------"
printf 'passed: %d   failed: %d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
echo "All container-log tests passed."
