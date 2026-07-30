#!/usr/bin/env bash
# Evidence that the Oryx build published usable output for trakt-ops-api.
#
#   bash deploy/trakt-ops-api/verify_oryx_output.sh <scm-host>
#
# THIS IS NOT THE DEPLOYMENT GATE. The definitive gate is the runtime check that
# follows it in the workflow: /health must return 200. This step exists to make a
# startup failure attributable, and it asserts only what the manifest itself says
# should be there.
#
# WHAT WAS WRONG BEFORE
# The previous version required `antenv/` or `__oryx_packages__/` to be readable
# over Kudu VFS and failed the job when neither was:
#
#     if antenv != 200 && __oryx_packages__ != 200; then fail
#
# That assumes the uncompressed layout. This service builds with
# --compress-destination-dir, and the build log ends:
#
#     --compress-destination-dir
#     Copied the compressed output to '/home/site/wwwroot'
#     Direct compression with zstd done
#     Manifest file created
#     Total execution done
#
# In that mode /home/site/wwwroot holds a single compressed artefact plus
# oryx-manifest.toml, and the runtime extracts it at startup. No expanded
# directory is published, so the assertion reported a successful build — 91
# packages installed, including pyyaml==6.0.3, deployment complete=true,
# status=4 (Success) — as a hard failure.
#
# Now: the manifest must exist (it is what the runtime needs), and whatever the
# manifest declares must exist. `antenv/` and `__oryx_packages__/` are reported
# as observations, never required.
#
# VERIFY_MANIFEST_TEXT and VERIFY_PROBE_RESULTS substitute the two Kudu reads so
# the tests cover every branch without an Azure subscription.
set -uo pipefail

SCM_HOST="${1:?usage: verify_oryx_output.sh <scm-host>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# VERIFY_FIXTURES is the switch, not the presence of a fixture value: "no
# manifest and nothing on the site" is a case the tests must be able to express,
# and both fixture variables are legitimately empty for it.
if [ -n "${VERIFY_FIXTURES:-}" ]; then
  # "path=code path=code ..." — a fixture for the tests. Anything not listed is
  # treated as 404, so a test only states what exists.
  probe() {
    local want="$1" pair
    for pair in $VERIFY_PROBE_RESULTS; do
      [ "${pair%%=*}" = "$want" ] && { printf '%s' "${pair#*=}"; return; }
    done
    printf '404'
  }
  fetch() { printf '%s' "${VERIFY_MANIFEST_TEXT:-}" > "$2"; [ -s "$2" ] && printf '200' || printf '404'; }
else
  TOKEN="$(az account get-access-token --resource https://management.azure.com/ \
            --query accessToken -o tsv 2>/dev/null)"
  if [ -z "${TOKEN:-}" ]; then
    echo "FAIL: could not acquire an access token for Kudu"
    exit 1
  fi
  # `-w '%{http_code}'` already prints 000 on a transport failure; do not append a
  # `|| echo 000` fallback, which is how a doubled `HTTP 000000` was produced.
  probe() {
    curl -sS -o /dev/null -w '%{http_code}' --max-time 60 \
      -H "Authorization: Bearer $TOKEN" \
      "https://${SCM_HOST}/api/vfs/site/wwwroot/$1" 2>/dev/null
  }
  fetch() {
    curl -sS -o "$2" -w '%{http_code}' --max-time 60 \
      -H "Authorization: Bearer $TOKEN" \
      "https://${SCM_HOST}/api/vfs/site/wwwroot/$1" 2>/dev/null
  }
fi

echo "=== Oryx build output in /home/site/wwwroot ==="

MANIFEST="$WORK/oryx-manifest.toml"
CODE="$(fetch oryx-manifest.toml "$MANIFEST")"
echo "  oryx-manifest.toml -> HTTP ${CODE:-000}"
if [ "${CODE:-000}" != "200" ]; then
  echo
  echo "FAIL: oryx-manifest.toml is not present. The runtime needs it to locate"
  echo "      the build output, so without it the worker cannot import its"
  echo "      dependencies. Read the Oryx build output in the deployment log to"
  echo "      see how far the build got; the diagnostics step prints it."
  exit 1
fi

echo "  --- manifest ---"
sed 's/^/    /' "$MANIFEST"

# Ask the manifest what to look for instead of assuming a layout.
PLAN="$WORK/plan"
if ! python3 "$HERE/oryx_manifest.py" plan --manifest "$MANIFEST" > "$PLAN"; then
  echo "FAIL: could not interpret oryx-manifest.toml"
  exit 1
fi

MODE=""; VENV=""; PKGDIR=""
CANDIDATES=()
while IFS='=' read -r key value; do
  case "$key" in
    mode) MODE="$value" ;;
    venv) VENV="$value" ;;
    pkgdir) PKGDIR="$value" ;;
    candidate) [ -n "$value" ] && CANDIDATES+=("$value") ;;
  esac
done < "$PLAN"

echo "  build mode declared by the manifest: ${MODE:-unknown}"
[ -n "$VENV" ]   && echo "  virtual env name  : $VENV"
[ -n "$PKGDIR" ] && echo "  package directory : $PKGDIR"

# Reported, never required. In compressed mode these legitimately do not exist.
echo "  --- observations (not requirements) ---"
for dir in "antenv/" "__oryx_packages__/" ${VENV:+"$VENV/"} ${PKGDIR:+"$PKGDIR/"}; do
  echo "    $dir -> HTTP $(probe "$dir")"
done

case "$MODE" in
compressed)
  echo "  --- compressed artefact declared by the manifest ---"
  FOUND=""
  for candidate in ${CANDIDATES+"${CANDIDATES[@]}"}; do
    CCODE="$(probe "$candidate")"
    echo "    $candidate -> HTTP $CCODE"
    if [ "$CCODE" = "200" ] && [ -z "$FOUND" ]; then
      FOUND="$candidate"
    fi
  done
  if [ -n "$FOUND" ]; then
    echo "  OK: manifest present and its compressed output ($FOUND) is on the site."
    echo "      The runtime extracts it at startup, so no expanded antenv/ or"
    echo "      __oryx_packages__/ directory is expected here."
    exit 0
  fi
  echo
  echo "FAIL: the manifest records compressed output but none of the candidate"
  echo "      artefacts is readable. Either the copy to /home/site/wwwroot did not"
  echo "      complete, or the manifest names an artefact this check did not"
  echo "      derive. Compare the names above with the 'Copied the compressed"
  echo "      output to' line in the Oryx build log, which the diagnostics step"
  echo "      prints, and widen oryx_manifest.py if the name is simply new."
  exit 1
  ;;

virtualenv)
  echo "  --- expanded environment declared by the manifest ---"
  OK=0
  for dir in ${VENV:+"$VENV/"} ${PKGDIR:+"$PKGDIR/"}; do
    DCODE="$(probe "$dir")"
    echo "    $dir -> HTTP $DCODE"
    [ "$DCODE" = "200" ] && OK=1
  done
  if [ "$OK" -eq 1 ]; then
    echo "  OK: manifest present and the expanded environment is on the site."
    exit 0
  fi
  echo
  echo "FAIL: the manifest declares an uncompressed environment"
  echo "      (${VENV:-?}/ ${PKGDIR:+, $PKGDIR/}) but it is not readable, so the"
  echo "      worker has no dependencies to import. Read the Oryx build output in"
  echo "      the deployment log; the diagnostics step prints it."
  exit 1
  ;;

*)
  # The manifest exists but records neither layout in a form this script knows.
  # Do not fail on that: the runtime health check that follows is the gate, and it
  # tests the thing that actually matters.
  echo
  echo "NOTE: oryx-manifest.toml is present but does not declare a build layout"
  echo "      this check recognises, so there is nothing specific to assert here."
  echo "      Not failing: the /health gate that follows decides the deployment."
  exit 0
  ;;
esac
