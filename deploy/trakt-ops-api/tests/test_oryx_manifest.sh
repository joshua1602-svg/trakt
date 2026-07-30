#!/usr/bin/env bash
# Behaviour tests for verify_oryx_output.sh and oryx_manifest.py.
#
# The case that matters most: the compressed build mode this service actually
# uses. The previous gate required antenv/ or __oryx_packages__/ over VFS and
# failed a build that had installed 91 packages (including pyyaml==6.0.3) and
# completed with status=4. These tests pin the corrected behaviour so that
# cannot come back.
#
# VERIFY_FIXTURES=1 selects the fixture path. VERIFY_MANIFEST_TEXT supplies the
# manifest; VERIFY_PROBE_RESULTS lists the
# paths that exist ("path=code ..."), everything else answering 404. No Azure.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$HERE/verify_oryx_output.sh"
PLANNER="$HERE/oryx_manifest.py"
PASS=0
FAIL=0

# The real shape for --compress-destination-dir with zstd.
COMPRESSED_MANIFEST='operationid="a1b2c3"
oryxversion="0.2.20250101.1"
PlatformName="python"
PlatformVersion="3.11.9"
compressDestinationDir="true"
compressedOutputFile="output.tar.zst"'

# Compression flagged, artefact NOT named — the key name varies by Oryx version,
# so the well-known names must still be probed.
COMPRESSED_UNNAMED='operationid="a1b2c3"
PlatformName="python"
compressDestinationDir="true"'

# The classic uncompressed layout.
VENV_MANIFEST='operationid="d4e5f6"
PlatformName="python"
PlatformVersion="3.11.9"
VirtualEnvName="antenv"
packagedir="__oryx_packages__"'

# expect <label> <expected-exit> <needle> <manifest> <probe-results>
expect() {
  local label="$1" want="$2" needle="$3" manifest="$4" probes="$5"
  local output rc
  output="$(VERIFY_FIXTURES=1 VERIFY_MANIFEST_TEXT="$manifest" VERIFY_PROBE_RESULTS="$probes" \
            bash "$SCRIPT" scm.example.net 2>&1)"
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

echo "--- compressed build mode (what this service actually does) ---"

# THE REGRESSION TEST. Manifest + output.tar.zst present, antenv and
# __oryx_packages__ absent. This is a PASS.
expect "compressed output with no antenv passes" 0 \
  "OK: manifest present and its compressed output (output.tar.zst)" \
  "$COMPRESSED_MANIFEST" \
  "oryx-manifest.toml=200 output.tar.zst=200"

expect "it says why no expanded directory is expected" 0 \
  "no expanded antenv/ or" \
  "$COMPRESSED_MANIFEST" \
  "oryx-manifest.toml=200 output.tar.zst=200"

expect "antenv/ absence is reported as an observation, not a requirement" 0 \
  "observations (not requirements)" \
  "$COMPRESSED_MANIFEST" \
  "oryx-manifest.toml=200 output.tar.zst=200"

expect "compression flagged without a filename still resolves output.tar.zst" 0 \
  "OK: manifest present and its compressed output (output.tar.zst)" \
  "$COMPRESSED_UNNAMED" \
  "oryx-manifest.toml=200 output.tar.zst=200"

expect "a gzip-compressed destination resolves too" 0 \
  "output.tar.gz" \
  "$COMPRESSED_UNNAMED" \
  "oryx-manifest.toml=200 output.tar.gz=200"

expect "a compressed virtualenv artefact resolves" 0 \
  "OK: manifest present and its compressed output (antenv.tar.gz)" \
  'PlatformName="python"
VirtualEnvName="antenv"
compressedvirtualenvfile="antenv.tar.gz"' \
  "oryx-manifest.toml=200 antenv.tar.gz=200"

# The genuine failure in compressed mode: nothing was copied to wwwroot.
expect "compressed mode with no artefact at all fails" 1 \
  "none of the candidate" \
  "$COMPRESSED_MANIFEST" \
  "oryx-manifest.toml=200"

expect "and it points at the Oryx log line that names the artefact" 1 \
  "Copied the compressed" \
  "$COMPRESSED_MANIFEST" \
  "oryx-manifest.toml=200"

echo "--- uncompressed build mode ---"

expect "expanded antenv passes" 0 \
  "the expanded environment is on the site" \
  "$VENV_MANIFEST" \
  "oryx-manifest.toml=200 antenv/=200"

expect "__oryx_packages__ alone passes" 0 \
  "the expanded environment is on the site" \
  "$VENV_MANIFEST" \
  "oryx-manifest.toml=200 __oryx_packages__/=200"

expect "declared but missing environment fails" 1 \
  "declares an uncompressed environment" \
  "$VENV_MANIFEST" \
  "oryx-manifest.toml=200"

echo "--- the manifest itself ---"

expect "a missing manifest fails (requirement retained)" 1 \
  "oryx-manifest.toml is not present" \
  "" \
  ""

expect "the manifest contents are printed for the record" 0 \
  "compressedOutputFile" \
  "$COMPRESSED_MANIFEST" \
  "oryx-manifest.toml=200 output.tar.zst=200"

# A manifest we cannot interpret must not fail the job — the runtime gate decides.
expect "an unrecognised layout defers to the /health gate" 0 \
  "Not failing: the /health gate" \
  'operationid="x"
PlatformName="python"' \
  "oryx-manifest.toml=200"

echo "--- no stale mid-flight-kill narrative ---"
for probes in "oryx-manifest.toml=200" "oryx-manifest.toml=200 output.tar.zst=200"; do
  out="$(VERIFY_FIXTURES=1 VERIFY_MANIFEST_TEXT="$COMPRESSED_MANIFEST" VERIFY_PROBE_RESULTS="$probes" \
         bash "$SCRIPT" scm.example.net 2>&1)"
  if printf '%s' "$out" | grep -qiF "killed mid-flight"; then
    echo "FAIL: output still claims the build was killed mid-flight"
    FAIL=$((FAIL + 1))
  else
    echo "pass: no mid-flight-kill claim ($probes)"
    PASS=$((PASS + 1))
  fi
done

echo "--- planner unit checks ---"
plan_says() {
  local label="$1" manifest="$2" needle="$3" tmp
  tmp="$(mktemp)"; printf '%s' "$manifest" > "$tmp"
  if python3 "$PLANNER" plan --manifest "$tmp" | grep -qFx -- "$needle"; then
    echo "pass: $label"; PASS=$((PASS + 1))
  else
    echo "FAIL: $label — plan did not emit '$needle'"
    python3 "$PLANNER" plan --manifest "$tmp" | sed 's/^/      | /'
    FAIL=$((FAIL + 1))
  fi
  rm -f "$tmp"
}

plan_says "compressed manifest -> mode=compressed" "$COMPRESSED_MANIFEST" "mode=compressed"
plan_says "compressed manifest names its artefact" "$COMPRESSED_MANIFEST" "candidate=output.tar.zst"
plan_says "venv manifest -> mode=virtualenv" "$VENV_MANIFEST" "mode=virtualenv"
plan_says "venv name is extracted" "$VENV_MANIFEST" "venv=antenv"
plan_says "package dir is extracted" "$VENV_MANIFEST" "pkgdir=__oryx_packages__"
plan_says "bare manifest -> mode=unknown" 'operationid="x"' "mode=unknown"
# Oryx capitalisation varies between versions; lookup must not depend on it.
plan_says "lowercase virtualenvname is understood" 'virtualenvname="antenv"' "venv=antenv"
# An empty compress flag says nothing and must not force compressed mode.
plan_says "an empty compress value does not imply compression" \
  'compressDestinationDir=""
VirtualEnvName="antenv"' "mode=virtualenv"

echo
echo "passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]
